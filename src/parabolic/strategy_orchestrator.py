"""Frugal historical-strategy campaign sidecar.

One scheduler tick advances at most one campaign iteration and performs at most
one model call. It only queues Parabolic's existing stdin backtests.
"""
from __future__ import annotations

import argparse, json, os, random, sqlite3, subprocess, hashlib, time, tempfile
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests

from parabolic.platform import RunStore, start_worker
from parabolic.strategy_runtime import validate_strategy_source

DEFAULT_ARTIFACT_REPOSITORY = Path("/Users/crivero/Documents/parabolic-strategies")
DEFAULT_STATE_DIR = Path(".parabolic/orchestrator")
STRATEGY_CONTRACT_PATH = Path(__file__).resolve().parents[2] / "docs" / "ORCHESTRATOR_STRATEGY_AUTHORING.md"

@dataclass(frozen=True)
class Campaign:
    campaign_id: str; symbol: str; model: str; status: str; iteration: int

def artifact_repository() -> Path:
    return Path(os.getenv("PARABOLIC_STRATEGY_REPOSITORY", str(DEFAULT_ARTIFACT_REPOSITORY)))

def codex_home(root: Path) -> Path:
    """Create isolated Codex provider configuration without persisting secrets."""
    home = root / ".parabolic" / "orchestrator" / "codex-home"; home.mkdir(parents=True, exist_ok=True)
    config = home / "config.toml"
    config.write_text("""model_provider = \"openrouter\"

[model_providers.openrouter]
name = \"OpenRouter\"
base_url = \"https://openrouter.ai/api/v1\"
env_key = \"OPENROUTER_API_KEY\"
wire_api = \"responses\"
request_max_retries = 2
stream_max_retries = 2
""", encoding="utf-8")
    return home

class Store:
    def __init__(self, path: Path = DEFAULT_STATE_DIR):
        path.mkdir(parents=True, exist_ok=True); self.db = sqlite3.connect(path / "campaigns.sqlite3"); self.db.row_factory = sqlite3.Row
        self.db.execute("CREATE TABLE IF NOT EXISTS campaigns (id TEXT PRIMARY KEY, symbol TEXT NOT NULL, model TEXT NOT NULL, status TEXT NOT NULL, iteration INTEGER NOT NULL DEFAULT 0, pending_run_id TEXT, parent_source TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS iterations (campaign_id TEXT, number INTEGER, run_id TEXT, source_path TEXT, source_hash TEXT, metadata TEXT, result TEXT, PRIMARY KEY(campaign_id, number))"); self.db.commit()
    def close(self): self.db.close()
    def create(self, campaign_id: str, symbol: str, model: str) -> Campaign:
        self.db.execute("INSERT INTO campaigns(id,symbol,model,status) VALUES(?,?,?, 'active')", (campaign_id,symbol,model)); self.db.commit(); return Campaign(campaign_id,symbol,model,"active",0)
    def active(self) -> list[Campaign]: return [Campaign(r["id"],r["symbol"],r["model"],r["status"],r["iteration"]) for r in self.db.execute("SELECT * FROM campaigns WHERE status='active'")]
    def row(self, cid: str): return self.db.execute("SELECT * FROM campaigns WHERE id=?",(cid,)).fetchone()
    def update(self,cid: str, **fields: object):
        sql=", ".join(f"{key}=?" for key in fields); self.db.execute(f"UPDATE campaigns SET {sql} WHERE id=?", (*fields.values(),cid)); self.db.commit()
    def add_iteration(self,cid: str,n:int,run_id:str,path:Path,source:str,metadata:dict[str,Any]):
        self.db.execute("INSERT INTO iterations VALUES(?,?,?,?,?,?,NULL)",(cid,n,run_id,str(path),hashlib.sha256(source.encode()).hexdigest(),json.dumps(metadata,sort_keys=True))); self.db.commit()
    def finish_iteration(self,cid:str,n:int,result:dict[str,Any]): self.db.execute("UPDATE iterations SET result=? WHERE campaign_id=? AND number=?",(json.dumps(result,sort_keys=True),cid,n)); self.db.commit()

def random_window(today: date | None=None) -> tuple[str,str]:
    today=today or date.today(); end=today-timedelta(days=random.randint(2, 365)); start=end-timedelta(days=random.randint(20,90)); return start.isoformat(),end.isoformat()

def consistency_score(result: dict[str,Any]) -> float:
    sharpe=float(result.get("sharpe") or 0); sortino=float(result.get("sortino") or 0); calmar=float(result.get("calmar") or 0); drawdown=abs(float(result.get("max_drawdown") or 0))
    return 0.20*sharpe+0.45*sortino+0.35*calmar-0.50*drawdown

def evidence() -> list[dict[str,Any]]:
    store=RunStore(Path(".parabolic/platform"))
    try:
        runs=[asdict(run) for run in store.list(100) if run.status=="succeeded" and run.result]
    finally: store.close()
    return sorted(runs,key=lambda run: consistency_score(run["result"]),reverse=True)[:6]

def compile_prompt(campaign: Campaign, prior_source: str | None, prior_results: list[dict[str,Any]]) -> tuple[str,dict[str,Any]]:
    # The contract is immutable mandatory context; evidence is the bounded variable context.
    compact=[{"run_id":r["run_id"],"result":r["result"],"metadata":r["metadata"]} for r in prior_results[:6]]
    contract=STRATEGY_CONTRACT_PATH.read_text(encoding="utf-8")
    prompt="\n\n".join((
        "You are one iteration of a historical strategy-improvement campaign. Do not discuss live trading.",
        "AUTHORITATIVE STRATEGY RUNTIME CONTRACT\n" + contract,
        "OUTPUT CONTRACT\nReturn one JSON object with exactly these keys: hypothesis, changes_from_parent, expected_effect, strategy_source, next_experiment_if_unsuccessful, metadata. strategy_source must contain only one valid Python function def strategy(ctx): ...; no Markdown or explanation belongs inside it.",
        f"CAMPAIGN\nSymbol: {campaign.symbol}\nIteration: {campaign.iteration + 1}\nObjective: maximize stable risk-adjusted returns (Sortino/Calmar first; positive Sharpe; low drawdown and stable daily returns).",
        "PARENT STRATEGY\n" + (prior_source or "No parent strategy; produce a simple safe baseline."),
        "DETERMINISTIC HISTORICAL EVIDENCE\n" + json.dumps(compact, sort_keys=True),
        "Make a single, falsifiable improvement. Preserve useful parent behavior unless the evidence contradicts it.",
    ))
    return prompt, {"selected_result_count":len(compact),"compiler":"deterministic-bounded-v2","strategy_contract_path":str(STRATEGY_CONTRACT_PATH),"strategy_contract_sha256":hashlib.sha256(contract.encode()).hexdigest()}

OUTPUT_SCHEMA={"type":"object","properties":{"hypothesis":{"type":"string"},"changes_from_parent":{"type":"array","items":{"type":"string"}},"expected_effect":{"type":"string"},"strategy_source":{"type":"string"},"next_experiment_if_unsuccessful":{"type":"string"},"metadata":{"type":"object"}},"required":["hypothesis","changes_from_parent","expected_effect","strategy_source","next_experiment_if_unsuccessful","metadata"],"additionalProperties":False}

def codex(prompt:str, model:str | None, root:Path) -> tuple[dict[str,Any],dict[str,Any]]:
    """Run one complete, ephemeral Codex turn and retain its JSONL telemetry."""
    with tempfile.TemporaryDirectory(prefix="parabolic-codex-") as temp:
        schema=Path(temp)/"schema.json"; output=Path(temp)/"result.json"; schema.write_text(json.dumps(OUTPUT_SCHEMA))
        command=["codex","exec","--ephemeral","--sandbox","read-only","--json","--output-schema",str(schema),"--output-last-message",str(output)]
        if model: command.extend(["--model",model])
        command.append(prompt)
        environment=os.environ.copy(); environment["CODEX_HOME"]=str(codex_home(root))
        completed=subprocess.run(command,cwd=root,text=True,capture_output=True,timeout=900,env=environment)
        events=[]
        for line in completed.stdout.splitlines():
            try: events.append(json.loads(line))
            except json.JSONDecodeError: continue
        if completed.returncode != 0: raise RuntimeError(f"Codex failed: {completed.stderr[-2000:]}")
        result=json.loads(output.read_text())
        started=next((event for event in events if event.get("type")=="thread.started"),{})
        completed_event=next((event for event in reversed(events) if event.get("type")=="turn.completed"),{})
        return result,{"provider":"codex","thread_id":started.get("thread_id"),"usage":completed_event.get("usage",{}),"event_count":len(events),"model":model}

def commit_artifact(campaign_id:str, iteration:int, source:str, manifest:dict[str,Any]) -> Path:
    repo=artifact_repository(); repo.mkdir(parents=True,exist_ok=True)
    if not (repo/".git").exists(): subprocess.run(["git","init"],cwd=repo,check=True,capture_output=True)
    path=repo/"strategies"/campaign_id/f"iteration-{iteration:02d}"; path.mkdir(parents=True,exist_ok=True); (path/"strategy.py").write_text(source); (path/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True))
    subprocess.run(["git","add",str(path.relative_to(repo))],cwd=repo,check=True); subprocess.run(["git","commit","-m",f"strategy: {campaign_id} iteration {iteration:02d}"],cwd=repo,check=True,capture_output=True)
    return path/"strategy.py"

def advance(store:Store,campaign:Campaign,root:Path) -> str:
    row=store.row(campaign.campaign_id)
    if row["pending_run_id"]:
        runs=RunStore(Path(".parabolic/platform")); run=runs.get(row["pending_run_id"]); runs.close()
        if run and run.status in {"queued","running"}: return "waiting"
        if run: store.finish_iteration(campaign.campaign_id,campaign.iteration,run.result or {"error":run.stderr}); store.update(campaign.campaign_id,pending_run_id=None)
        if campaign.iteration>=4: store.update(campaign.campaign_id,status="completed"); return "completed"
    n=campaign.iteration+1; prompt,report=compile_prompt(campaign,row["parent_source"],evidence()); generated,usage=codex(prompt,campaign.model,root)
    required={"hypothesis","changes_from_parent","expected_effect","strategy_source","next_experiment_if_unsuccessful","metadata"}
    missing=required-set(generated)
    if missing: raise ValueError(f"agent response is missing required fields: {', '.join(sorted(missing))}")
    source=str(generated["strategy_source"]); validate_strategy_source(source)
    metadata={"campaign_id":campaign.campaign_id,"iteration":n,"hypothesis":generated["hypothesis"],"changes_from_parent":generated["changes_from_parent"],"expected_effect":generated["expected_effect"],"next_experiment_if_unsuccessful":generated["next_experiment_if_unsuccessful"],"agent_metadata":generated["metadata"],"compiler_report":report,"codex":usage,"strategy_sha256":hashlib.sha256(source.encode()).hexdigest()}
    artifact=commit_artifact(campaign.campaign_id,n,source,metadata); start,end=random_window(); run_state=Path(".parabolic/platform"); start_worker(run_state,root); runs=RunStore(run_state); run=runs.queue(["evaluate","--symbol",campaign.symbol,"--start",start,"--end",end,"--timeframe","minute","--strategy-stdin"],source,root,metadata|{"strategy_artifact_path":str(artifact)}); runs.close(); store.add_iteration(campaign.campaign_id,n,run.run_id,artifact,source,metadata); store.update(campaign.campaign_id,iteration=n,pending_run_id=run.run_id,parent_source=source); return "queued"

def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description="Historical Parabolic strategy orchestrator")
    parser.add_argument("command",choices=["create","tick","serve"]); parser.add_argument("--campaign-id"); parser.add_argument("--symbol",default="SPY"); parser.add_argument("--model",default=None); parser.add_argument("--root",default=str(Path.cwd()))
    args=parser.parse_args(argv); store=Store()
    try:
        if args.command=="create":
            if not args.campaign_id: parser.error("create requires --campaign-id")
            print(json.dumps(asdict(store.create(args.campaign_id,args.symbol,args.model or ""))))
        else:
            while True:
                results={campaign.campaign_id:advance(store,campaign,Path(args.root)) for campaign in store.active()}; print(json.dumps(results))
                if args.command=="tick": break
                time.sleep(300)
    finally: store.close()
    return 0

if __name__ == "__main__": raise SystemExit(main())
