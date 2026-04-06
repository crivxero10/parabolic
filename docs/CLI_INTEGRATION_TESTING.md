# CLI Integration Testing

This document describes the fixture-backed CLI integration tests for the `evaluate` and `tune` workflows, including the stdin strategy path for `evaluate`, and how to reproduce them exactly when they fail.

## Purpose

The tests exercise the real `driver.py` argument parser and the `evaluate` and `tune` workflows without hitting Alpaca or depending on live credentials.

It is meant to catch breakage in:

- CLI argument parsing
- strategy resolution
- market-data gathering and timestamp alignment
- backtest/evaluate wiring
- summary emission

It is not meant to validate live market data or the current behavior of production strategies.

## Stable Inputs

The test uses:

- A checked-in frozen market-data fixture at [evaluate_fixture.json](/Users/mini-c/Trading/parabolic/tests/fixtures/evaluate_fixture.json)
- A checked-in frozen market-data fixture at [tune_fixture.json](/Users/mini-c/Trading/parabolic/tests/fixtures/tune_fixture.json)
- A deterministic strategy named `deterministic_test`
- A deterministic tuning strategy named `deterministic_tune`
- A fixture-backed market-data provider defined in [test_driver_cli.py](/Users/mini-c/Trading/parabolic/tests/test_driver_cli.py)

Because the provider and data are frozen, the expected output is stable and suitable for regression testing.

## Expected Output Shape

The `evaluate` integration test expects the CLI to emit JSON containing at least these fields:

- `strategy_name`
- `parameters`
- `sharpe`
- `sortino`
- `calmar`
- `max_drawdown`
- `final_balance`
- `pnl_pct_all_time`
- `pnl_pct_daily_average`
- `best_day`
- `worst_day`
- `log_file`

The current expected values are asserted in [test_driver_cli.py](/Users/mini-c/Trading/parabolic/tests/test_driver_cli.py).

There is also an `evaluate` stdin-strategy test that submits Python source defining:

```python
def strategy(ctx):
    ...
```

That path is expected to emit the same summary shape as the normal built-in evaluate flow when the strategy succeeds.

When stdin strategy validation or execution fails, the CLI is expected to emit a JSON error payload with fields like:

- `error_type`
- `message`
- `phase`
- optional `traceback`
- optional `captured_stdout`
- optional `captured_stderr`
- optional `context`

The `tune` integration test expects the CLI to emit JSON containing these sections:

- `best_sharpe`
- `best_sortino`
- `best_calmar`
- `best_final_balance`
- `best_composite`

For the deterministic tune fixture, every section should point to the same optimal parameter set:

```json
{
  "k_st": 5,
  "k_lt": 120,
  "crab_upper_bound": 1.5
}
```

The current expected values are asserted in [test_driver_cli.py](/Users/mini-c/Trading/parabolic/tests/test_driver_cli.py).

## Repro Steps

Run the integration test only:

```bash
pytest tests/test_driver_cli.py -q
```

Run the full suite:

```bash
pytest -q
```

Reproduce the exact CLI execution path manually from Python:

```python
import io
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from parabolic.driver import main
from parabolic.mdp import MarketDataProvider


class FixtureBackedMarketDataProvider(MarketDataProvider):
    def __init__(self, fixture_path: Path):
        self._payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    def get_bars(self, symbol, timeframe, start, end, adjustment="all", feed=None):
        del timeframe, adjustment, feed
        start_day = start[:10]
        end_day = end[:10]
        return [
            dict(bar)
            for bar in self._payload.get(symbol, [])
            if start_day <= str(bar["t"])[:10] <= end_day
        ]

    def get_latest_bar(self, symbol, feed=None):
        del feed
        bars = self._payload.get(symbol, [])
        return dict(bars[-1]) if bars else None

fixture_path = Path("tests/fixtures/evaluate_fixture.json")
provider = FixtureBackedMarketDataProvider(fixture_path)
stdout = io.StringIO()
stderr = io.StringIO()

with redirect_stdout(stdout), redirect_stderr(stderr):
    exit_code = main(
        [
            "evaluate",
            "--symbol", "SPY",
            "--start", "2024-01-08",
            "--end", "2024-01-11",
            "--timeframe", "minute",
            "--strategy-name", "deterministic_test",
            "--k-st", "6",
            "--k-lt", "42",
            "--lookback", "11",
            "--crab-lower-bound", "-2",
            "--crab-upper-bound", "2",
            "--rolling-stop-pct", "-0.1",
        ],
        market_data_provider_override=provider,
    )

print(exit_code)
print(stderr.getvalue())
print(json.loads(stdout.getvalue()))
```

Reproduce the exact `tune` CLI execution path manually from Python:

```python
import io
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from parabolic.driver import main
from parabolic.mdp import MarketDataProvider


class FixtureBackedMarketDataProvider(MarketDataProvider):
    def __init__(self, fixture_path: Path):
        self._payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    def get_bars(self, symbol, timeframe, start, end, adjustment="all", feed=None):
        del timeframe, adjustment, feed
        start_day = start[:10]
        end_day = end[:10]
        return [
            dict(bar)
            for bar in self._payload.get(symbol, [])
            if start_day <= str(bar["t"])[:10] <= end_day
        ]

    def get_latest_bar(self, symbol, feed=None):
        del feed
        bars = self._payload.get(symbol, [])
        return dict(bars[-1]) if bars else None

fixture_path = Path("tests/fixtures/tune_fixture.json")
provider = FixtureBackedMarketDataProvider(fixture_path)
stdout = io.StringIO()
stderr = io.StringIO()

with redirect_stdout(stdout), redirect_stderr(stderr):
    exit_code = main(
        [
            "tune",
            "--symbol", "SPY",
            "--start", "2024-02-05",
            "--end", "2024-02-08",
            "--timeframe", "minute",
            "--strategy-name", "deterministic_tune",
        ],
        market_data_provider_override=provider,
    )

print(exit_code)
print(stderr.getvalue())
print(json.loads(stdout.getvalue()))
```

Reproduce the stdin strategy `evaluate` path manually from Python:

```python
import io
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from parabolic.driver import main
from parabolic.mdp import MarketDataProvider


class FixtureBackedMarketDataProvider(MarketDataProvider):
    def __init__(self, fixture_path: Path):
        self._payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    def get_bars(self, symbol, timeframe, start, end, adjustment="all", feed=None):
        del timeframe, adjustment, feed
        start_day = start[:10]
        end_day = end[:10]
        return [
            dict(bar)
            for bar in self._payload.get(symbol, [])
            if start_day <= str(bar["t"])[:10] <= end_day
        ]

    def get_latest_bar(self, symbol, feed=None):
        del feed
        bars = self._payload.get(symbol, [])
        return dict(bars[-1]) if bars else None

fixture_path = Path("tests/fixtures/evaluate_fixture.json")
provider = FixtureBackedMarketDataProvider(fixture_path)
strategy_source = '''
def strategy(ctx):
    current_price = float(ctx.market[ctx.t][ctx.asset_name])
    current_units = int(ctx.brokerage.positions.get(ctx.asset_name, 0))
    timestamp = ctx.bar["t"] if ctx.bar is not None else None
    if ctx.t == 1 and current_units == 0:
        units = min(10, int(ctx.brokerage.available_cash // current_price))
        if units > 0:
            ctx.brokerage.execute(ctx.asset_name, units, current_price, timestamp=timestamp)
        return
    if ctx.is_session_end and current_units > 0:
        ctx.brokerage.execute(ctx.asset_name, -current_units, current_price, timestamp=timestamp)
'''

stdout = io.StringIO()
stderr = io.StringIO()

with redirect_stdout(stdout), redirect_stderr(stderr):
    exit_code = main(
        [
            "evaluate",
            "--symbol", "SPY",
            "--start", "2024-01-08",
            "--end", "2024-01-11",
            "--timeframe", "minute",
            "--strategy-stdin",
        ],
        market_data_provider_override=provider,
        stdin_override=strategy_source,
    )

print(exit_code)
print(stderr.getvalue())
print(json.loads(stdout.getvalue()))
```

## Troubleshooting

If one of these tests fails:

1. Confirm the fixture file did not change unexpectedly.
2. Confirm the deterministic strategy logic in [driver.py](/Users/mini-c/Trading/parabolic/src/parabolic/driver.py) did not change.
3. If the stdin strategy path failed, check whether the source violated runtime validation rules such as imports, `while` loops, or forbidden builtin calls.
4. Re-run the manual reproduction snippet above and inspect the emitted JSON.
5. If the output changed intentionally, update the expected values in [test_driver_cli.py](/Users/mini-c/Trading/parabolic/tests/test_driver_cli.py) only after verifying the new values are correct.

## Design Notes

These tests intentionally use deterministic strategies instead of `regime_classifier`.

That keeps the tests focused on whether the CLI plumbing works end to end. A failure here should indicate broken integration, not just that a production strategy evolved.
