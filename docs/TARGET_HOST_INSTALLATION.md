# Target Host Installation and Operations Runbook

This runbook is for installing Parabolic on a new host and proving each layer
works. Do not report the system as fully operational until the applicable gates
below pass.

The tested baseline is commit `a874c2f` on `main`.

## Status vocabulary

Use these statuses precisely:

- **Installed**: source, dependencies, tests, and offline smoke checks pass.
- **Platform ready**: the local worker starts, reports status, and the dashboard
  responds.
- **MCP ready**: an MCP client can connect and complete a queue/get-run cycle.
- **Backtest ready**: an authenticated Alpaca-backed evaluation completes.
- **Orchestrator ready**: OpenRouter/Codex can generate a strategy, the strategy
  is evaluated, and the artifact/result linkage is visible.
- **Production-compliant**: all controls in
  [ORCHESTRATOR_AGENT_CONTRACT.md](/Users/crivero/Documents/parabolic/parabolic/docs/ORCHESTRATOR_AGENT_CONTRACT.md)
  are mechanically enforced. The current revision is not production-compliant;
  its contract documents remaining technical debt.

Never upgrade one status to another implicitly.

## Gate 0: Host prerequisites

The host needs:

- Git;
- `uv`;
- Python `>=3.14` available through `uv`;
- Codex CLI available as `codex` for orchestrator runs;
- network access to GitHub and Alpaca;
- OpenRouter access for orchestrator runs; and
- a writable directory for local state, market-data cache, logs, and strategy
  artifacts.

Run:

```bash
command -v git
command -v uv
command -v codex
uv run python --version
codex --version
```

If `codex` is not installed, report **Installed** only. Do not report
**Orchestrator ready**.

## Gate 1: Clone and dependency installation

Choose a stable host directory. The artifact repository must be separate from
the source repository.

```bash
export PARABOLIC_ROOT="/Users/mini-c/parabolic"
git clone https://github.com/crivxero10/parabolic.git "$PARABOLIC_ROOT"
cd "$PARABOLIC_ROOT"
git checkout main
git rev-parse HEAD
uv sync
```

The revision must be `a874c2f` or a later approved `main` revision. If the
checkout already exists, use `git fetch origin` and verify the requested revision
without discarding local changes.

Set an explicit host-specific artifact path. Do not use the development machine's
default path:

```bash
export PARABOLIC_STRATEGY_REPOSITORY="$PARABOLIC_ROOT/.parabolic-strategies"
mkdir -p "$PARABOLIC_STRATEGY_REPOSITORY"
```

## Gate 2: Offline verification

Run the suite using the repository's supported runner:

```bash
cd "$PARABOLIC_ROOT"
uv run python -m unittest discover -s tests
```

Expected result:

```text
Ran 128 tests
OK
```

Run the no-credential smoke checks:

```bash
uv run python -m parabolic.driver strategy-spec
uv run python -m parabolic.strategy_orchestrator --help
uv run python -m parabolic.platform status
uv run python -m compileall -q src tests
```

Before continuing, record:

- Git revision;
- test count and result;
- Python and Codex versions;
- host OS and timezone; and
- artifact repository path.

At this point the correct status is **Installed**, not **Platform ready**.

## Gate 3: Secret configuration

Alpaca credentials are required for real market-data backtests. OpenRouter is
required only for Codex-driven strategy generation. MCP and the local platform can
be tested without either key.

Inject secrets through the host secret manager or a non-versioned, permission-
restricted environment. Never put them in CLI arguments, MCP inputs, prompts,
SQLite, logs, Git, or this document.

Required for backtests:

```text
ALPACA_API_KEY
ALPACA_API_SECRET
```

Required for orchestrator runs:

```text
OPENROUTER_API_KEY
```

The required orchestrator model is:

```text
deepseek/deepseek-v4-flash
```

After injection, verify only presence, never print values:

```bash
test -n "${ALPACA_API_KEY:-}" && echo "ALPACA_API_KEY present"
test -n "${ALPACA_API_SECRET:-}" && echo "ALPACA_API_SECRET present"
test -n "${OPENROUTER_API_KEY:-}" && echo "OPENROUTER_API_KEY present"
```

## Gate 4: Start and verify the local platform

From the repository root:

```bash
uv run python -m parabolic.platform start
uv run python -m parabolic.platform status
```

Expected status:

```json
{"running": true, "state_dir": ".parabolic/platform"}
```

The platform worker is a local process. Record its state directory and do not
edit `.parabolic/platform/` directly.

Start the dashboard in a separate terminal:

```bash
cd "$PARABOLIC_ROOT"
uv run python -m parabolic.platform web
```

In another terminal, verify the HTTP surface:

```bash
curl --fail --silent http://127.0.0.1:8765/api/runs
```

The dashboard is localhost-only. Do not bind it to a network interface.

## Gate 5: Verify MCP queueing

Configure the MCP client with:

```json
{
  "mcpServers": {
    "parabolic": {
      "command": "uv",
      "args": ["run", "python", "-m", "parabolic.mcp_server"],
      "cwd": "/Users/mini-c/parabolic"
    }
  }
}
```

The target agent must perform this exact sequence:

1. Call `get_platform_spec`.
2. Call `queue_run` with `{"argv":["agent-spec"]}`.
3. Save the returned `run_id` and repository revision.
4. Poll `get_run(run_id)` until `succeeded`, `failed`, or `cancelled`.
5. Confirm the run appears in `list_runs` and `/api/runs`.
6. Preserve stdout, stderr, exit code, status, metadata, and run ID.

This is the **MCP ready** gate. It does not prove Alpaca or OpenRouter access.

## Gate 6: Real backtest smoke test

Only run this gate after Alpaca credentials are present:

```bash
cd "$PARABOLIC_ROOT"
uv run python -m parabolic.driver evaluate \
  --symbol SPY \
  --start 2024-01-08 \
  --end 2024-01-11 \
  --timeframe minute \
  --strategy-name deterministic_test \
  --k-st 6 \
  --k-lt 42 \
  --lookback 11 \
  --crab-lower-bound -2 \
  --crab-upper-bound 2 \
  --rolling-stop-pct -0.1
```

Pass criteria:

- exit code is zero;
- stdout is valid JSON;
- the result contains risk metrics and final balance;
- no credential appears in stdout or stderr;
- the market-data cache is populated; and
- the run is visible in the dashboard.

If Alpaca returns no data, an alignment error, a rate-limit error, or a partial
session, report **Platform ready** but not **Backtest ready**.

## Gate 7: Codex/OpenRouter preflight

Only run this gate after `codex` is installed and `OPENROUTER_API_KEY` is present.
Confirm that the key is read from the environment and that the configured model
is exactly `deepseek/deepseek-v4-flash`. Do not write the key to Codex config.

The preflight must verify:

- Codex can start with the isolated `CODEX_HOME` configuration;
- OpenRouter Responses API access succeeds;
- structured JSON output succeeds;
- the model/provider identity is recorded; and
- usage, duration, and failure telemetry are retained without secrets.

If preflight fails, stop before creating an autonomous campaign. Do not retry
indefinitely and do not substitute an unapproved model.

## Gate 8: Controlled orchestrator smoke test

This gate creates an actual strategy artifact and may consume model/API budget.
Use a unique host-local campaign ID and explicitly pass the model:

```bash
export CAMPAIGN_ID="host-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
uv run python -m parabolic.strategy_orchestrator create \
  --campaign-id "$CAMPAIGN_ID" \
  --symbol SPY \
  --model deepseek/deepseek-v4-flash \
  --root "$PARABOLIC_ROOT"
```

Do not run `tick` until Gates 4, 6, and 7 pass. A tick may invoke Codex, call
Alpaca, queue a backtest, and create a Git artifact:

```bash
uv run python -m parabolic.strategy_orchestrator tick \
  --model deepseek/deepseek-v4-flash \
  --root "$PARABOLIC_ROOT"
```

Confirm that:

- one Codex generation was recorded;
- the generated source passed validation;
- a strategy artifact was committed under
  `$PARABOLIC_STRATEGY_REPOSITORY`;
- a platform `run_id` exists;
- the result or structured failure is persisted; and
- the source and result are visible to a human for cross-checking.

Do not start `serve` unattended until the operator accepts the documented
technical debt in the autonomous-run contract. The current implementation does
not yet mechanically enforce every target requirement, including persisted
30-minute scheduling, immutable archives, exact latest-30-session resolution,
parallel-run leases, and all budget controls.

## Final handoff report

The installing agent must report:

```text
Host path:
Git revision:
Python version:
uv version:
Codex version:
Artifact repository:
Tests:
Offline smoke checks:
Platform status:
Dashboard status:
MCP status:
Alpaca backtest status:
OpenRouter preflight status:
Orchestrator smoke status:
Secrets configured: yes/no (never include values)
Known blockers:
```

The agent must not say “fully installed and operational” when any applicable gate
is skipped. It must state exactly which gate was not run and why.
