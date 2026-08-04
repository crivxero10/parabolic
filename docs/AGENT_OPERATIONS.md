# Agent Operations Guide

Use Parabolic through its MCP server for all asynchronous work. Do not import
trading modules, open the SQLite database, or invoke the worker directly. The
server queues the established public CLI and records a run identifier, repository
snapshot, status, output, and error stream for every request.

## Connect

Register this stdio server with an MCP client from the repository root:

```json
{
  "mcpServers": {
    "parabolic": {
      "command": "uv",
      "args": ["run", "python", "-m", "parabolic.mcp_server"],
      "cwd": "/Users/crivero/Documents/parabolic/parabolic"
    }
  }
}
```

The MCP server starts the worker automatically when the first run is queued. For
human-operated lifecycle control, use the single command family:

```bash
uv run python -m parabolic.platform start
uv run python -m parabolic.platform status
uv run python -m parabolic.platform stop
```

## Required Agent Workflow

1. Call `get_platform_spec` before an unfamiliar task to read the component and
   evidence contract.
2. Call `queue_run` with an argument vector for a public CLI command.
3. Save the returned `run_id` and `revision`; they are the durable correlation
   keys for the request.
4. Poll `get_run(run_id)` until the status is `succeeded`, `failed`, or
   `cancelled`. Use `list_runs` only to discover or recover run IDs.
5. Treat `stdout`, `stderr`, `exit_code`, and the recorded repository snapshot
   as the result evidence. Report failures with the run ID and command.
6. Use `cancel_run` only while a run is `queued`; running-process cancellation
   is not implemented yet.

## Queueing Commands

`queue_run` accepts every existing CLI command as an `argv` list. The command
name is the first item; do not include `python -m parabolic.driver`.

```json
{"argv": ["agent-spec"]}
```

```json
{
  "argv": [
    "evaluate", "--symbol", "SPY", "--start", "2024-01-08", "--end", "2024-01-11",
    "--timeframe", "minute", "--strategy-name", "regime_classifier",
    "--k-st", "6", "--k-lt", "42", "--lookback", "11",
    "--crab-lower-bound", "-2", "--crab-upper-bound", "2", "--rolling-stop-pct", "-0.1"
  ]
}
```

For stdin strategies, include `--strategy-stdin` in `argv` and send the exact
Python source in `stdin_source`. The strategy source remains subject to the
existing guarded runtime described in [STRATEGY_API.md](/Users/crivero/Documents/parabolic/parabolic/docs/STRATEGY_API.md).

## Credentials and Safety

Workers inherit `ALPACA_API_KEY` and `ALPACA_API_SECRET` from their existing
environment. Never send credentials in MCP tool inputs: `--api-key`,
`--api-secret`, and their `--flag=value` forms are rejected and must not be
retried. The local run database is `.parabolic/platform/runs.sqlite3`; it is not
an API and must not be edited by agents.

## Failure Handling

- `failed`: inspect `stderr`, then `stdout`; preserve the `run_id` in any task
  evidence.
- `queued` for an unexpectedly long time: call `platform_control` with
  `{"action": "status"}`. Start it with `{"action": "start"}` if needed.
- `cancelled`: queue a new run rather than modifying the cancelled record.
- Unknown run: recover it with `list_runs`; never guess a run ID.

Known platform limitations and future work are kept in
[PLATFORM_OPERATIONS.md](/Users/crivero/Documents/parabolic/parabolic/docs/PLATFORM_OPERATIONS.md).
