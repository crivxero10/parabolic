# Platform Operations

The sidecar platform lets agents queue any public Parabolic CLI command without
importing trading-domain modules. The worker invokes `python -m parabolic.driver`
with the queued arguments, so `evaluate`, `tune`, `strategy-spec`, and
`agent-spec` retain their existing behavior.

Agents should follow [AGENT_OPERATIONS.md](/Users/crivero/Documents/parabolic/parabolic/docs/AGENT_OPERATIONS.md) for the required MCP workflow.

## One Lifecycle Switch

Use one command family to control the local worker:

```bash
uv run python -m parabolic.platform start
uv run python -m parabolic.platform status
uv run python -m parabolic.platform stop
```

The first `queue_run` MCP call also starts the worker if it is not already
running. Local state lives in `.parabolic/platform/` and is intentionally ignored
by Git. Credentials are never stored in a run record; workers inherit the
existing `ALPACA_API_KEY` and `ALPACA_API_SECRET` environment pattern.

## MCP Server

Run the stdio server from the repository root:

```bash
uv run python -m parabolic.mcp_server
```

It exposes `queue_run`, `get_run`, `list_runs`, `cancel_run`,
`platform_control`, and `get_platform_spec`. `queue_run` accepts the CLI argument
vector, for example `{"argv": ["agent-spec"]}` or an `evaluate`/`tune` argument
vector. `--api-key` and `--api-secret` are rejected to keep secrets out of the
SQLite queue.

Each run stores agent-provided JSON metadata alongside its CLI output and parsed
JSON result. Start the localhost dashboard with:

```bash
uv run python -m parabolic.platform web
```

It serves `http://127.0.0.1:8765` with a strategy-results table and CRUD API:
`GET /api/runs`, `GET /api/runs/{run_id}`, `PATCH /api/runs/{run_id}` for metadata,
and `DELETE /api/runs/{run_id}` for terminal records.

## Deferred Technical Debt

- The worker uses a local SQLite database and a PID file; it is single-host and
  has no cross-machine leases or leader election.
- Cancellation is immediate only for queued work. Cancelling a running CLI
  subprocess requires process-group tracking and a cooperative cancellation API.
- Runs retain raw stdout/stderr in SQLite without retention limits or redaction.
- The MCP transport is stdio for local agents. A network deployment needs
  streamable HTTP, authentication, authorization, and an external secrets store.
- The daemon currently validates command shape, not every CLI semantic. The
  existing CLI remains the authoritative validation boundary.
- The dashboard is deliberately local-only and has no authentication. Do not
  bind it to a network interface until access control is added.
