# Orchestrator Design: Strategy Campaigns

The normative behavior for agents and the daemon is defined in
[ORCHESTRATOR_AGENT_CONTRACT.md](/Users/crivero/Documents/parabolic/parabolic/docs/ORCHESTRATOR_AGENT_CONTRACT.md).
Where this design summary or the current implementation differs from that
contract, the contract wins and the difference is technical debt—not permission
for an agent to improvise.

The strategy orchestrator is a sidecar. It has no live-trading capability and
does not import Parabolic trading modules. It uses Parabolic MCP to queue
historical `evaluate --strategy-stdin` runs, then reads the same durable results
that the dashboard displays.

## Configuration and Operation

The orchestrator wraps `codex exec` for one complete, ephemeral, read-only Codex
run per strategy iteration. It configures Codex with OpenRouter's Responses API
provider and reads `OPENROUTER_API_KEY` only from the process environment; the
key is never written to disk. The required default model is
`deepseek/deepseek-v4-flash`. Its JSONL thread and usage telemetry are stored with
strategy metadata. The current prototype exposes manual campaign commands:

```bash
uv run python -m parabolic.strategy_orchestrator create --campaign-id momentum-v1 --symbol SPY
uv run python -m parabolic.strategy_orchestrator tick
uv run python -m parabolic.strategy_orchestrator serve
```

## Target Run Policy

- Scheduler cadence: one persisted, idempotent slot every 30 minutes.
- Strategy iterations per independent run: exactly four, sequentially.
- Evaluation window: one frozen set of the latest 30 fully completed US equity
  trading sessions shared by all four iterations.
- Different runs may execute concurrently up to a default limit of two. Full
  slots are recorded as skipped instead of accumulating a backlog.
- Objective: consistent risk-adjusted returns. Rank candidates by a deterministic
  composite emphasizing Sortino and Calmar, requiring positive Sharpe and
  penalizing max drawdown and unstable daily returns.
- Live orders, broker APIs, and real-money execution are out of scope.

## Strategy Artifact Repository

The orchestrator owns a separate local Git repository. Its default path is
`/Users/crivero/Documents/parabolic-strategies`; `PARABOLIC_STRATEGY_REPOSITORY`
may override it. It must not be nested in the Parabolic source repository. A
generated strategy is written and committed before queueing its backtest, under:

```text
strategies/<run-id>/iteration-<01..04>/
  strategy.py
  manifest.json
  compiled-context-report.json
  evaluation-result.json
```

`manifest.json` links strategy hash, parent strategy, campaign/cycle/iteration,
OpenRouter model and usage, deterministic prompt-compiler report, and Parabolic
`run_id`. The dashboard reads this metadata and provides a source link/path so a
human can cross-check the exact evaluated code.

## Frugal Agent Loop

The orchestrator turns Parabolic results into deterministic aggregates and passes
only selected, allowlisted evidence to its lightweight prompt compiler. Exactly
one complete Codex execution produces the plan, source, and metadata for each
iteration; that execution may itself consume multiple provider requests, so hard
token and cost budgets still apply. The sidecar then validates source, commits
the artifact, queues the run, and stores results. It archives the independent run
after four iterations or a terminal budget/infrastructure failure.
