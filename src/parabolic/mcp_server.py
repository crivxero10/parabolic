"""MCP adapter for the Parabolic sidecar run platform."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from parabolic.agent_native import project_spec
from parabolic.platform import (
    DEFAULT_STATE_DIR,
    RunStore,
    start_worker,
    state_dir_path,
    stop_worker,
    validate_cli_argv,
)


ROOT = Path.cwd()
mcp = FastMCP("Parabolic", instructions="Queue and inspect Parabolic CLI runs. Credentials stay in the worker environment.", json_response=True)


def _store(state_dir: str | None) -> RunStore:
    return RunStore(state_dir_path(state_dir) if state_dir else DEFAULT_STATE_DIR)


@mcp.tool()
def queue_run(argv: list[str], stdin_source: str | None = None, metadata: dict[str, object] | None = None, state_dir: str | None = None) -> dict[str, object]:
    """Queue any supported Parabolic CLI command; workers inherit existing Alpaca environment credentials."""
    directory = state_dir_path(state_dir) if state_dir else DEFAULT_STATE_DIR
    validate_cli_argv(argv)
    start_worker(directory, ROOT)
    store = _store(state_dir)
    try:
        return asdict(store.queue(argv, stdin_source, ROOT, metadata))
    finally:
        store.close()


@mcp.tool()
def get_run(run_id: str, state_dir: str | None = None) -> dict[str, object] | None:
    """Return state, output, and repository snapshot for one queued run."""
    store = _store(state_dir)
    try:
        run = store.get(run_id)
        return asdict(run) if run else None
    finally:
        store.close()


@mcp.tool()
def list_runs(limit: int = 50, state_dir: str | None = None) -> list[dict[str, object]]:
    """List recent platform runs, newest first."""
    store = _store(state_dir)
    try:
        return [asdict(run) for run in store.list(max(1, min(limit, 200)))]
    finally:
        store.close()


@mcp.tool()
def cancel_run(run_id: str, state_dir: str | None = None) -> dict[str, object] | None:
    """Cancel a queued run. Running subprocess cancellation is deferred technical debt."""
    store = _store(state_dir)
    try:
        run = store.cancel(run_id)
        return asdict(run) if run else None
    finally:
        store.close()


@mcp.tool()
def platform_control(action: str, state_dir: str | None = None) -> dict[str, object]:
    """Use the single lifecycle switch: start, stop, or status."""
    directory = state_dir_path(state_dir) if state_dir else DEFAULT_STATE_DIR
    if action == "start":
        return {"started": start_worker(directory, ROOT), "state_dir": str(directory)}
    if action == "stop":
        return {"stopped": stop_worker(directory)}
    if action == "status":
        from parabolic.platform import worker_running
        return {"running": worker_running(directory), "state_dir": str(directory)}
    raise ValueError("action must be start, stop, or status")


@mcp.tool()
def get_platform_spec() -> dict[str, object]:
    """Return the durable architecture and task-packet contract."""
    return project_spec()


if __name__ == "__main__":
    mcp.run()
