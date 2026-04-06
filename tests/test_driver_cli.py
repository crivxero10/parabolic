import io
import json
import math
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from parabolic.driver import main
from parabolic.mdp import MarketDataProvider


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "evaluate_fixture.json"
TUNE_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tune_fixture.json"


class FixtureBackedMarketDataProvider(MarketDataProvider):
    def __init__(self, fixture_path: Path):
        self._payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        adjustment: str = "all",
        feed: str | None = None,
    ) -> list[dict]:
        del timeframe, adjustment, feed
        start_day = start[:10]
        end_day = end[:10]
        return [
            dict(bar)
            for bar in self._payload.get(symbol, [])
            if start_day <= str(bar["t"])[:10] <= end_day
        ]

    def get_latest_bar(self, symbol: str, feed: str | None = None) -> dict | None:
        del feed
        bars = self._payload.get(symbol, [])
        if not bars:
            return None
        return dict(bars[-1])


class TestDriverCliEvaluate(unittest.TestCase):
    def test_evaluate_cli_returns_stable_summary_for_fixture_backed_provider(self):
        provider = FixtureBackedMarketDataProvider(FIXTURE_PATH)
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "evaluate",
                    "--symbol",
                    "SPY",
                    "--start",
                    "2024-01-08",
                    "--end",
                    "2024-01-11",
                    "--timeframe",
                    "minute",
                    "--strategy-name",
                    "deterministic_test",
                    "--k-st",
                    "6",
                    "--k-lt",
                    "42",
                    "--lookback",
                    "11",
                    "--crab-lower-bound",
                    "-2",
                    "--crab-upper-bound",
                    "2",
                    "--rolling-stop-pct",
                    "-0.1",
                ],
                market_data_provider_override=provider,
            )

        assert exit_code == 0
        assert stderr.getvalue() == ""

        payload = json.loads(stdout.getvalue())
        assert payload["strategy_name"] == "deterministic_test"
        assert payload["parameters"] == {
            "k_st": 6,
            "k_lt": 42,
            "lookback": 11,
            "crab_lower_bound": -2.0,
            "crab_upper_bound": 2.0,
            "rolling_stop_pct": -0.1,
        }
        assert math.isclose(payload["sharpe"], 3.0693868916990183, rel_tol=1e-12)
        assert math.isclose(payload["sortino"], 7.948163741808041, rel_tol=1e-12)
        assert math.isclose(payload["calmar"], 80.1994517405211, rel_tol=1e-12)
        assert math.isclose(payload["max_drawdown"], -0.0028903593602785937, rel_tol=1e-12)
        assert math.isclose(payload["final_balance"], 10033.146999999999, rel_tol=1e-12)
        assert math.isclose(payload["pnl_pct_all_time"], 0.3314700000000004, rel_tol=1e-12)
        assert math.isclose(payload["pnl_pct_daily_average"], 0.0834628060768361, rel_tol=1e-12)
        assert payload["best_day"] == {
            "date": "2024-01-10",
            "pnl_pct": 0.5051041297199329,
        }
        assert payload["worst_day"] == {
            "date": "2024-01-09",
            "pnl_pct": -0.2890359360278612,
        }
        assert str(payload["log_file"]).endswith(".log")

    def test_evaluate_cli_accepts_strategy_from_stdin(self):
        provider = FixtureBackedMarketDataProvider(FIXTURE_PATH)
        stdout = io.StringIO()
        stderr = io.StringIO()
        strategy_source = """
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
"""

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "evaluate",
                    "--symbol",
                    "SPY",
                    "--start",
                    "2024-01-08",
                    "--end",
                    "2024-01-11",
                    "--timeframe",
                    "minute",
                    "--strategy-stdin",
                ],
                market_data_provider_override=provider,
                stdin_override=strategy_source,
            )

        assert exit_code == 0
        assert stderr.getvalue() == ""

        payload = json.loads(stdout.getvalue())
        assert payload["strategy_name"] == "strategy_stdin"
        assert payload["parameters"] == {}
        assert math.isclose(payload["sharpe"], 3.0693868916990183, rel_tol=1e-12)
        assert math.isclose(payload["final_balance"], 10033.146999999999, rel_tol=1e-12)
        assert str(payload["log_file"]).endswith(".log")

    def test_evaluate_cli_rejects_disallowed_strategy_source(self):
        provider = FixtureBackedMarketDataProvider(FIXTURE_PATH)
        stdout = io.StringIO()
        stderr = io.StringIO()
        strategy_source = """
def strategy(ctx):
    while True:
        pass
"""

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "evaluate",
                    "--symbol",
                    "SPY",
                    "--start",
                    "2024-01-08",
                    "--end",
                    "2024-01-11",
                    "--timeframe",
                    "minute",
                    "--strategy-stdin",
                ],
                market_data_provider_override=provider,
                stdin_override=strategy_source,
            )

        assert exit_code == 1
        assert stderr.getvalue() == ""
        payload = json.loads(stdout.getvalue())
        assert payload["error_type"] == "StrategyValidationError"
        assert payload["phase"] == "strategy_validation"
        assert "While" in payload["message"]

    def test_evaluate_cli_times_out_long_running_strategy_source(self):
        provider = FixtureBackedMarketDataProvider(FIXTURE_PATH)
        stdout = io.StringIO()
        stderr = io.StringIO()
        strategy_source = """
def strategy(ctx):
    for _ in range(10 ** 8):
        pass
"""

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "evaluate",
                    "--symbol",
                    "SPY",
                    "--start",
                    "2024-01-08",
                    "--end",
                    "2024-01-11",
                    "--timeframe",
                    "minute",
                    "--strategy-stdin",
                    "--strategy-timeout-seconds",
                    "0.2",
                ],
                market_data_provider_override=provider,
                stdin_override=strategy_source,
            )

        assert exit_code == 1
        assert stderr.getvalue() == ""
        payload = json.loads(stdout.getvalue())
        assert payload["error_type"] == "StrategyTimeout"
        assert payload["phase"] == "strategy_timeout"
        assert payload["context"]["timeout_seconds"] == 0.2


class TestDriverCliTune(unittest.TestCase):
    def test_tune_cli_returns_known_optimal_parameter_set_for_fixture_backed_provider(self):
        provider = FixtureBackedMarketDataProvider(TUNE_FIXTURE_PATH)
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "tune",
                    "--symbol",
                    "SPY",
                    "--start",
                    "2024-02-05",
                    "--end",
                    "2024-02-08",
                    "--timeframe",
                    "minute",
                    "--strategy-name",
                    "deterministic_tune",
                ],
                market_data_provider_override=provider,
            )

        assert exit_code == 0
        assert stderr.getvalue() == ""

        payload = json.loads(stdout.getvalue())
        expected_parameters = {
            "k_st": 5,
            "k_lt": 120,
            "crab_upper_bound": 1.5,
        }

        for section_name in (
            "best_sharpe",
            "best_sortino",
            "best_calmar",
            "best_final_balance",
            "best_composite",
        ):
            assert section_name in payload
            section = payload[section_name]
            assert section["strategy_name"] == "deterministic_tune"
            assert section["parameters"] == expected_parameters
            assert section["ranks"] == {
                "sharpe": 1,
                "sortino": 1,
                "calmar": 1,
                "final_balance": 1,
            }

        assert math.isclose(payload["best_final_balance"]["final_balance"], 12072.136000000002, rel_tol=1e-12)
