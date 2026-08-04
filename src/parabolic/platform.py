"""Sidecar queue and worker for agent-submitted CLI runs.

This module is intentionally an adapter: workers invoke the existing public CLI
instead of importing or changing trading-domain code.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_STATE_DIR = Path(".parabolic/platform")
ALLOWED_COMMANDS = {"tune", "evaluate", "strategy-spec", "agent-spec"}
SECRET_FLAGS = {"--api-key", "--api-secret"}


@dataclass(frozen=True, slots=True)
class Run:
    run_id: str
    argv: list[str]
    stdin_source: str | None
    status: str
    revision: str
    dirty: bool
    exit_code: int | None
    stdout: str
    stderr: str


def state_dir_path(value: str | Path | None = None) -> Path:
    return Path(value) if value is not None else DEFAULT_STATE_DIR


def validate_cli_argv(argv: Sequence[str]) -> list[str]:
    normalized = [str(value) for value in argv]
    if not normalized or normalized[0] not in ALLOWED_COMMANDS:
        raise ValueError(f"first argument must be one of: {', '.join(sorted(ALLOWED_COMMANDS))}")
    if any(value in SECRET_FLAGS or value.startswith(("--api-key=", "--api-secret=")) for value in normalized):
        raise ValueError("credentials must be provided through ALPACA_API_KEY and ALPACA_API_SECRET")
    return normalized


def repository_snapshot(root: Path) -> tuple[str, bool]:
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "diff", "--quiet"], cwd=root, check=False).returncode != 0
    return revision, dirty


class RunStore:
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.state_dir / "runs.sqlite3")
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("""CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY, argv TEXT NOT NULL, stdin_source TEXT,
            status TEXT NOT NULL, revision TEXT NOT NULL, dirty INTEGER NOT NULL,
            exit_code INTEGER, stdout TEXT NOT NULL DEFAULT '', stderr TEXT NOT NULL DEFAULT ''
        )""")
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def queue(self, argv: Sequence[str], stdin_source: str | None, root: Path) -> Run:
        revision, dirty = repository_snapshot(root)
        run = Run(uuid.uuid4().hex, validate_cli_argv(argv), stdin_source, "queued", revision, dirty, None, "", "")
        self.connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run.run_id, json.dumps(run.argv), run.stdin_source, run.status, run.revision, run.dirty, None, "", ""),
        )
        self.connection.commit()
        return run

    def get(self, run_id: str) -> Run | None:
        row = self.connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return self._to_run(row) if row else None

    def list(self, limit: int = 50) -> list[Run]:
        rows = self.connection.execute("SELECT * FROM runs ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
        return [self._to_run(row) for row in rows]

    def claim(self) -> Run | None:
        self.connection.execute("BEGIN IMMEDIATE")
        row = self.connection.execute("SELECT * FROM runs WHERE status = 'queued' ORDER BY rowid LIMIT 1").fetchone()
        if row is None:
            self.connection.commit()
            return None
        self.connection.execute("UPDATE runs SET status = 'running' WHERE run_id = ?", (row["run_id"],))
        self.connection.commit()
        return self.get(row["run_id"])

    def finish(self, run_id: str, exit_code: int, stdout: str, stderr: str) -> None:
        status = "succeeded" if exit_code == 0 else "failed"
        self.connection.execute("UPDATE runs SET status=?, exit_code=?, stdout=?, stderr=? WHERE run_id=?", (status, exit_code, stdout, stderr, run_id))
        self.connection.commit()

    def cancel(self, run_id: str) -> Run | None:
        self.connection.execute("UPDATE runs SET status='cancelled' WHERE run_id=? AND status='queued'", (run_id,))
        self.connection.commit()
        return self.get(run_id)

    @staticmethod
    def _to_run(row: sqlite3.Row) -> Run:
        return Run(row["run_id"], json.loads(row["argv"]), row["stdin_source"], row["status"], row["revision"], bool(row["dirty"]), row["exit_code"], row["stdout"], row["stderr"])


def run_once(state_dir: Path, root: Path) -> Run | None:
    store = RunStore(state_dir)
    try:
        run = store.claim()
        if run is None:
            return None
        completed = subprocess.run([sys.executable, "-m", "parabolic.driver", *run.argv], cwd=root, input=run.stdin_source, text=True, capture_output=True, env=os.environ.copy())
        store.finish(run.run_id, completed.returncode, completed.stdout, completed.stderr)
        return store.get(run.run_id)
    finally:
        store.close()


def worker(state_dir: Path, root: Path, poll_seconds: float = 0.2) -> None:
    while True:
        if run_once(state_dir, root) is None:
            time.sleep(poll_seconds)


def _pid_path(state_dir: Path) -> Path:
    return state_dir / "worker.pid"


def worker_running(state_dir: Path) -> bool:
    try:
        pid = int(_pid_path(state_dir).read_text())
        os.kill(pid, 0)
        return True
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return False


def start_worker(state_dir: Path, root: Path) -> bool:
    state_dir.mkdir(parents=True, exist_ok=True)
    if worker_running(state_dir):
        return False
    process = subprocess.Popen([sys.executable, "-m", "parabolic.platform", "worker", "--state-dir", str(state_dir), "--root", str(root)], cwd=root, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _pid_path(state_dir).write_text(str(process.pid))
    return True


def stop_worker(state_dir: Path) -> bool:
    if not worker_running(state_dir):
        return False
    os.kill(int(_pid_path(state_dir).read_text()), signal.SIGTERM)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parabolic agent-run platform")
    parser.add_argument("command", choices=["start", "stop", "status", "worker", "run-once"])
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--root", default=str(Path.cwd()))
    args = parser.parse_args(argv)
    state_dir, root = state_dir_path(args.state_dir), Path(args.root).resolve()
    if args.command == "start":
        print(json.dumps({"started": start_worker(state_dir, root), "state_dir": str(state_dir)}))
    elif args.command == "stop":
        print(json.dumps({"stopped": stop_worker(state_dir)}))
    elif args.command == "status":
        print(json.dumps({"running": worker_running(state_dir), "state_dir": str(state_dir)}))
    elif args.command == "worker":
        worker(state_dir, root)
    else:
        run = run_once(state_dir, root)
        print(json.dumps(asdict(run) if run else {"status": "idle"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
