import json
import math
import unittest
from pathlib import Path

from parabolic.strategy_runtime import (
    StrategyRuntimeConfig,
    StrategyValidationError,
    _build_error_payload,
    _truncate,
    evaluate_strategy_source,
    load_strategy_from_source,
    validate_strategy_source,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "evaluate_fixture.json"


def _load_fixture_payload() -> dict[str, list[dict]]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _build_runtime_inputs() -> tuple[list[dict], list[dict[str, float]]]:
    payload = _load_fixture_payload()
    spy_bars = payload["SPY"]
    spxl_by_timestamp = {bar["t"]: float(bar["c"]) for bar in payload["SPXL"]}
    spxs_by_timestamp = {bar["t"]: float(bar["c"]) for bar in payload["SPXS"]}

    raw_bars: list[dict] = []
    snapshots: list[dict[str, float]] = []
    for bar in spy_bars:
        timestamp = bar["t"]
        raw_bars.append(dict(bar))
        snapshots.append(
            {
                "SPY": float(bar["c"]),
                "SPXL": spxl_by_timestamp[timestamp],
                "SPXS": spxs_by_timestamp[timestamp],
            }
        )
    return raw_bars, snapshots


class TestStrategyRuntimeValidation(unittest.TestCase):
    def test_validate_strategy_source_accepts_valid_strategy(self):
        validate_strategy_source(
            """
def strategy(ctx):
    return None
"""
        )

    def test_validate_strategy_source_rejects_imports(self):
        with self.assertRaises(StrategyValidationError):
            validate_strategy_source(
                """
def strategy(ctx):
    import os
    return os.getcwd()
"""
            )

    def test_validate_strategy_source_rejects_while_loops(self):
        with self.assertRaises(StrategyValidationError):
            validate_strategy_source(
                """
def strategy(ctx):
    while True:
        return None
"""
            )

    def test_validate_strategy_source_rejects_bad_signature(self):
        with self.assertRaises(StrategyValidationError):
            validate_strategy_source(
                """
def strategy(ctx, extra):
    return None
"""
            )

    def test_load_strategy_from_source_returns_callable(self):
        strategy = load_strategy_from_source(
            """
def strategy(ctx):
    return ctx.t
"""
        )

        assert callable(strategy)

    def test_load_strategy_from_source_exposes_indicators_without_import(self):
        strategy = load_strategy_from_source(
            """
def strategy(ctx):
    closes = [1.0, 2.0, 3.0]
    return Indicators.sma(closes, 2)[-1]
"""
        )

        class _Ctx:
            t = 0

        assert strategy(_Ctx()) == 2.5


class TestStrategyRuntimeHelpers(unittest.TestCase):
    def test_truncate_shortens_long_values(self):
        value = "x" * 20

        result = _truncate(value, 5)

        assert result == "xxxxx...<truncated>"

    def test_build_error_payload_includes_optional_fields(self):
        payload = _build_error_payload(
            error_type="ExampleError",
            message="bad things happened",
            phase="strategy_runtime",
            max_output_chars=100,
            stdout="hello",
            stderr="oops",
            tb="trace",
            context={"t": 5},
        )

        assert payload["error_type"] == "ExampleError"
        assert payload["captured_stdout"] == "hello"
        assert payload["captured_stderr"] == "oops"
        assert payload["traceback"] == "trace"
        assert payload["context"] == {"t": 5}


class TestEvaluateStrategySource(unittest.TestCase):
    def test_evaluate_strategy_source_returns_summary_for_valid_strategy(self):
        raw_bars, snapshots = _build_runtime_inputs()
        success, payload, exit_code = evaluate_strategy_source(
            strategy_source="""
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
""",
            symbol="SPY",
            raw_bars=raw_bars,
            snapshots=snapshots,
            start="2024-01-08",
            end="2024-01-11",
            initial_balance=10000.0,
            log_file="runtime.log",
        )

        assert success is True
        assert exit_code == 0
        assert payload["strategy_name"] == "strategy_stdin"
        assert math.isclose(payload["final_balance"], 10033.146999999999, rel_tol=1e-12)
        assert payload["log_file"] == "runtime.log"

    def test_evaluate_strategy_source_returns_validation_error_payload(self):
        raw_bars, snapshots = _build_runtime_inputs()
        success, payload, exit_code = evaluate_strategy_source(
            strategy_source="""
def strategy(ctx):
    import os
    return os.getcwd()
""",
            symbol="SPY",
            raw_bars=raw_bars,
            snapshots=snapshots,
            start="2024-01-08",
            end="2024-01-11",
            initial_balance=10000.0,
            log_file="runtime.log",
        )

        assert success is False
        assert exit_code == 1
        assert payload["error_type"] == "StrategyValidationError"
        assert payload["phase"] == "strategy_validation"

    def test_evaluate_strategy_source_returns_runtime_error_with_captured_output(self):
        raw_bars, snapshots = _build_runtime_inputs()
        success, payload, exit_code = evaluate_strategy_source(
            strategy_source="""
def strategy(ctx):
    print("debug line")
    return 1 / 0
""",
            symbol="SPY",
            raw_bars=raw_bars,
            snapshots=snapshots,
            start="2024-01-08",
            end="2024-01-11",
            initial_balance=10000.0,
            log_file="runtime.log",
        )

        assert success is False
        assert exit_code == 1
        assert payload["error_type"] == "ZeroDivisionError"
        assert payload["phase"] == "strategy_runtime"
        assert "debug line" in payload["captured_stdout"]
        assert "division by zero" in payload["traceback"]
        assert payload["context"]["asset_name"] == "SPY"
        assert payload["context"]["t"] >= 0
        assert "timestamp" in payload["context"]

    def test_evaluate_strategy_source_times_out_long_running_strategy(self):
        raw_bars, snapshots = _build_runtime_inputs()
        success, payload, exit_code = evaluate_strategy_source(
            strategy_source="""
def strategy(ctx):
    for _ in range(10 ** 8):
        pass
""",
            symbol="SPY",
            raw_bars=raw_bars,
            snapshots=snapshots,
            start="2024-01-08",
            end="2024-01-11",
            initial_balance=10000.0,
            log_file="runtime.log",
            config=StrategyRuntimeConfig(timeout_seconds=0.2),
        )

        assert success is False
        assert exit_code == 1
        assert payload["error_type"] == "StrategyTimeout"
        assert payload["phase"] == "strategy_timeout"
        assert payload["context"]["timeout_seconds"] == 0.2

    def test_evaluate_strategy_source_can_use_context_market_history(self):
        raw_bars, snapshots = _build_runtime_inputs()
        success, payload, exit_code = evaluate_strategy_source(
            strategy_source="""
def strategy(ctx):
    current_price = float(ctx.market[ctx.t][ctx.asset_name])
    current_units = int(ctx.brokerage.positions.get(ctx.asset_name, 0))
    closes = [float(row[ctx.asset_name]) for row in ctx.market]
    timestamp = ctx.bar["t"] if ctx.bar is not None else None
    if len(closes) >= 2 and closes[-1] > closes[-2] and current_units == 0:
        units = min(5, int(ctx.brokerage.available_cash // current_price))
        if units > 0:
            ctx.brokerage.execute(ctx.asset_name, units, current_price, timestamp=timestamp)
        return
    if ctx.is_session_end and current_units > 0:
        ctx.brokerage.execute(ctx.asset_name, -current_units, current_price, timestamp=timestamp)
""",
            symbol="SPY",
            raw_bars=raw_bars,
            snapshots=snapshots,
            start="2024-01-08",
            end="2024-01-11",
            initial_balance=10000.0,
            log_file="runtime.log",
        )

        assert success is True
        assert exit_code == 0
        assert payload["strategy_name"] == "strategy_stdin"
        assert payload["final_balance"] > 10000.0

    def test_evaluate_strategy_source_can_use_indicators_without_import(self):
        raw_bars, snapshots = _build_runtime_inputs()
        success, payload, exit_code = evaluate_strategy_source(
            strategy_source="""
def strategy(ctx):
    closes = [float(row[ctx.asset_name]) for row in ctx.market]
    current_units = int(ctx.brokerage.positions.get(ctx.asset_name, 0))
    timestamp = ctx.bar["t"] if ctx.bar is not None else None
    fast = Indicators.sma(closes, 2)[-1] if len(closes) >= 2 else None
    slow = Indicators.sma(closes, 3)[-1] if len(closes) >= 3 else None
    current_price = float(ctx.market[ctx.t][ctx.asset_name])
    if fast is not None and slow is not None and fast > slow and current_units == 0:
        units = min(4, int(ctx.brokerage.available_cash // current_price))
        if units > 0:
            ctx.brokerage.execute(ctx.asset_name, units, current_price, timestamp=timestamp)
        return
    if ctx.is_session_end and current_units > 0:
        ctx.brokerage.execute(ctx.asset_name, -current_units, current_price, timestamp=timestamp)
""",
            symbol="SPY",
            raw_bars=raw_bars,
            snapshots=snapshots,
            start="2024-01-08",
            end="2024-01-11",
            initial_balance=10000.0,
            log_file="runtime.log",
        )

        assert success is True
        assert exit_code == 0
        assert payload["strategy_name"] == "strategy_stdin"
        assert payload["final_balance"] > 10000.0
