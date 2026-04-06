# CLI Integration Testing

This document describes the fixture-backed CLI integration test for the `evaluate` workflow and how to reproduce it exactly when it fails.

## Purpose

The test exercises the real `driver.py` argument parser and `evaluate` workflow without hitting Alpaca or depending on live credentials.

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
- A deterministic strategy named `deterministic_test`
- A fixture-backed market-data provider defined in [test_driver_cli.py](/Users/mini-c/Trading/parabolic/tests/test_driver_cli.py)

Because the provider and data are frozen, the expected output is stable and suitable for regression testing.

## Expected Output Shape

The integration test expects the CLI to emit JSON containing at least these fields:

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

## Troubleshooting

If this test fails:

1. Confirm the fixture file did not change unexpectedly.
2. Confirm the deterministic strategy logic in [driver.py](/Users/mini-c/Trading/parabolic/src/parabolic/driver.py) did not change.
3. Re-run the manual reproduction snippet above and inspect the emitted JSON.
4. If the output changed intentionally, update the expected values in [test_driver_cli.py](/Users/mini-c/Trading/parabolic/tests/test_driver_cli.py) only after verifying the new values are correct.

## Design Notes

This test intentionally uses a deterministic strategy instead of `regime_classifier`.

That keeps the test focused on whether the CLI plumbing works end to end. A failure here should indicate broken integration, not just that a production strategy evolved.
