import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from parabolic.driver import main, parse_market_symbols, strategy_api_spec
from parabolic.mdp import MarketDataProvider


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "evaluate_fixture.json"


class FixtureBackedMarketDataProvider(MarketDataProvider):
    def __init__(self, payload: dict[str, list[dict]]):
        self._payload = payload

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


class DroppingSymbolProvider(FixtureBackedMarketDataProvider):
    def __init__(self, payload: dict[str, list[dict]], dropped_symbol: str):
        super().__init__(payload)
        self._dropped_symbol = dropped_symbol

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        adjustment: str = "all",
        feed: str | None = None,
    ) -> list[dict]:
        if symbol == self._dropped_symbol:
            return []
        return super().get_bars(symbol, timeframe, start, end, adjustment, feed)


def _load_payload_with_vixy() -> dict[str, list[dict]]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    payload["VIXY"] = []
    for index, bar in enumerate(payload["SPY"]):
        payload["VIXY"].append(
            {
                "t": bar["t"],
                "o": 20.0 + (index * 0.01),
                "h": 20.2 + (index * 0.01),
                "l": 19.8 + (index * 0.01),
                "c": 20.1 + (index * 0.01),
                "v": 5000 + index,
                "vw": 20.05 + (index * 0.01),
                "n": 50 + (index % 10),
            }
        )
    return payload


class TestDriverStrategyApi(unittest.TestCase):
    def test_parse_market_symbols_preserves_primary_and_dedupes(self):
        result = parse_market_symbols("SPY", "SPY,SPXL,SPXS,VIXY,SPY")

        assert result == ["SPY", "SPXL", "SPXS", "VIXY"]

    def test_parse_market_symbols_defaults_to_backward_compatible_universe(self):
        result = parse_market_symbols("SPY", None)

        assert result == ["SPY", "SPXL", "SPXS"]

    def test_strategy_api_spec_exposes_expected_contract_sections(self):
        spec = strategy_api_spec()

        assert spec["strategy_signature"] == "def strategy(ctx): ..."
        assert "ctx_fields" in spec
        assert "brokerage_methods" in spec
        assert "indicator_methods" in spec
        assert spec["market_symbols"]["override_flag"] == "--market-symbols"

    def test_strategy_spec_cli_emits_json_without_credentials(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(["strategy-spec"])

        assert exit_code == 0
        assert stderr.getvalue() == ""
        payload = json.loads(stdout.getvalue())
        assert payload["strategy_signature"] == "def strategy(ctx): ..."

    def test_market_symbols_flag_loads_declared_universe_into_ctx_market(self):
        provider = FixtureBackedMarketDataProvider(_load_payload_with_vixy())
        stdout = io.StringIO()
        stderr = io.StringIO()
        strategy_source = """
def strategy(ctx):
    required_symbols = ("SPY", "SPXL", "SPXS", "VIXY")
    for snapshot in ctx.market:
        for symbol in required_symbols:
            if symbol not in snapshot:
                raise ValueError(f"missing {symbol}")
    current_units = int(ctx.brokerage.positions.get("VIXY", 0))
    price = float(ctx.market[ctx.t]["VIXY"])
    timestamp = ctx.bar["t"] if ctx.bar is not None else None
    if ctx.t == 1 and current_units == 0:
        units = min(3, int(ctx.brokerage.available_cash // price))
        if units > 0:
            ctx.brokerage.execute("VIXY", units, price, timestamp=timestamp)
        return
    if ctx.is_session_end and current_units > 0:
        ctx.brokerage.execute("VIXY", -current_units, price, timestamp=timestamp)
"""

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "evaluate",
                    "--symbol",
                    "SPY",
                    "--market-symbols",
                    "SPY,SPXL,SPXS,VIXY",
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
        assert payload["final_balance"] is not None

    def test_market_symbols_cli_rejects_empty_list(self):
        provider = FixtureBackedMarketDataProvider(_load_payload_with_vixy())
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "evaluate",
                    "--symbol",
                    "SPY",
                    "--market-symbols",
                    ",,,",
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

        assert exit_code == 1
        assert stdout.getvalue() == ""
        assert "--market-symbols must contain at least one symbol" in stderr.getvalue()

    def test_market_symbols_cli_injects_primary_symbol_when_omitted(self):
        provider = FixtureBackedMarketDataProvider(_load_payload_with_vixy())
        stdout = io.StringIO()
        stderr = io.StringIO()
        strategy_source = """
def strategy(ctx):
    required_symbols = ("SPY", "SPXL", "SPXS", "VIXY")
    for snapshot in ctx.market:
        for symbol in required_symbols:
            if symbol not in snapshot:
                raise ValueError(f"missing {symbol}")
"""

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "evaluate",
                    "--symbol",
                    "SPY",
                    "--market-symbols",
                    "SPXL,SPXS,VIXY",
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

    def test_market_symbols_cli_fails_when_declared_symbol_has_no_data(self):
        provider = DroppingSymbolProvider(_load_payload_with_vixy(), "VIXY")
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "evaluate",
                    "--symbol",
                    "SPY",
                    "--market-symbols",
                    "SPY,SPXL,SPXS,VIXY",
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

        assert exit_code == 1
        assert stdout.getvalue() == ""
        assert "Finished fetching VIXY: no data returned." in stderr.getvalue()

    def test_market_symbols_stdin_strategy_can_use_indicators_without_import(self):
        provider = FixtureBackedMarketDataProvider(_load_payload_with_vixy())
        stdout = io.StringIO()
        stderr = io.StringIO()
        strategy_source = """
def strategy(ctx):
    closes = [float(row["VIXY"]) for row in ctx.market]
    timestamp = ctx.bar["t"] if ctx.bar is not None else None
    current_units = int(ctx.brokerage.positions.get("VIXY", 0))
    current_price = float(ctx.market[ctx.t]["VIXY"])
    fast = Indicators.sma(closes, 2)[-1] if len(closes) >= 2 else None
    slow = Indicators.sma(closes, 3)[-1] if len(closes) >= 3 else None
    if fast is not None and slow is not None and fast > slow and current_units == 0:
        units = min(2, int(ctx.brokerage.available_cash // current_price))
        if units > 0:
            ctx.brokerage.execute("VIXY", units, current_price, timestamp=timestamp)
        return
    if ctx.is_session_end and current_units > 0:
        ctx.brokerage.execute("VIXY", -current_units, current_price, timestamp=timestamp)
"""

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "evaluate",
                    "--symbol",
                    "SPY",
                    "--market-symbols",
                    "SPY,SPXL,SPXS,VIXY",
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
