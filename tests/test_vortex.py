import unittest
import math
import statistics

from parabolic.classifier import Regime
from parabolic.vortex import VortexGenerator, VortexRegimeBias, VortexRequest


def _minute_timestamp(day: str, minute: int) -> str:
    hour = 14 + ((30 + minute) // 60)
    minute_of_hour = (30 + minute) % 60
    return f"{day}T{hour:02d}:{minute_of_hour:02d}:00Z"


def _make_session(
    day: str,
    start_price: float,
    *,
    base_delta: float = 0.0006,
    direction_pattern: set[int] | None = None,
    wick_ratio: float = 0.0008,
    pattern_modulus: int = 7,
) -> list[dict]:
    bars = []
    current = start_price
    pattern = direction_pattern or {0, 1, 2, 5}
    for minute in range(390):
        open_price = current
        direction = 1 if minute % pattern_modulus in pattern else -1
        delta = base_delta * direction
        close_price = max(open_price * (1 + delta), 0.01)
        high_price = max(open_price, close_price) * (1.0 + wick_ratio)
        low_price = min(open_price, close_price) * (1.0 - wick_ratio)
        bars.append(
            {
                "t": _minute_timestamp(day, minute),
                "o": round(open_price, 4),
                "h": round(high_price, 4),
                "l": round(low_price, 4),
                "c": round(close_price, 4),
                "v": 1000 + minute,
                "vw": round((open_price + close_price) / 2, 4),
                "n": 100 + (minute % 11),
            }
        )
        current = close_price
    return bars


def _session_log_return(bars: list[dict]) -> float:
    return math.log(float(bars[-1]["c"]) / float(bars[0]["o"]))


def _minute_return_stdev(bars: list[dict]) -> float:
    minute_returns = [
        math.log(float(bar["c"]) / float(bar["o"]))
        for bar in bars
    ]
    return statistics.stdev(minute_returns)


class _FakeMarketDataProvider:
    def __init__(self):
        self._bars_by_range: list[tuple[str, str, str, str, list[dict]]] = []
        self.latest_bar = {"t": "2025-03-31T19:59:00Z"}

    def add_range(self, symbol: str, start: str, end: str, bars: list[dict]) -> None:
        self._bars_by_range.append((symbol, "1Min", start, end, bars))

    def get_bars(self, symbol: str, timeframe: str, start: str, end: str, adjustment: str = "all", feed: str | None = None):
        for stored_symbol, stored_timeframe, stored_start, stored_end, bars in self._bars_by_range:
            if (
                stored_symbol == symbol
                and stored_timeframe == timeframe
                and stored_start <= start
                and stored_end >= end
            ):
                return list(bars)
        return []

    def get_latest_bar(self, symbol: str, feed: str | None = None):
        return self.latest_bar


class TestVortexRequest(unittest.TestCase):
    def test_request_requires_length_for_dateless_generation(self):
        with self.assertRaises(ValueError):
            VortexRequest(symbol="SPY", regime=Regime.BULL)

    def test_request_validates_partial_date_ranges(self):
        with self.assertRaises(ValueError):
            VortexRequest(symbol="SPY", regime=Regime.BULL, start_date="2024-01-01")

    def test_request_normalizes_bias(self):
        request = VortexRequest(symbol="spy", regime="bull", length_days=2, regime_bias="strong")

        assert request.symbol == "SPY"
        assert request.regime == Regime.BULL
        assert request.regime_bias == VortexRegimeBias.STRONG


class TestVortexGenerator(unittest.TestCase):
    def _build_provider_with_reference_sessions(self, sessions: list[tuple[str, float, float, set[int] | None]]):
        provider = _FakeMarketDataProvider()
        reference_bars = []
        for day, price, delta, pattern in sessions:
            reference_bars.extend(
                _make_session(
                    day,
                    price,
                    base_delta=delta,
                    direction_pattern=pattern,
                )
            )
        provider.add_range("SPY", "2020-03-01", "2025-03-30", reference_bars)
        return provider

    def test_generate_dateless_vortex_is_seeded_and_returns_full_sessions(self):
        provider = self._build_provider_with_reference_sessions(
            [
                ("2025-03-24", 500.0, 0.0006, {0, 1, 2, 5}),
                ("2025-03-25", 504.0, 0.0006, {0, 1, 2, 5}),
                ("2025-03-26", 501.0, 0.0006, {0, 1, 2, 5}),
                ("2025-03-27", 507.0, 0.0006, {0, 1, 2, 5}),
            ]
        )

        generator = VortexGenerator(provider)
        request = VortexRequest(
            symbol="SPY",
            regime=Regime.BULL,
            length_days=2,
            seed=7,
            regime_bias=VortexRegimeBias.STRONG,
        )

        vortex_one = generator.generate(request)
        vortex_two = generator.generate(request)

        assert len(vortex_one.bars) == 780
        assert vortex_one.bars == vortex_two.bars
        assert vortex_one.bars[0]["t"].startswith("3000-01-03T14:30:00Z")
        assert all(set(bar) == {"t", "o", "h", "l", "c", "v", "vw", "n"} for bar in vortex_one.bars)

    def test_generated_bars_preserve_ohlcv_invariants(self):
        provider = self._build_provider_with_reference_sessions(
            [
                ("2025-03-24", 500.0, 0.0006, {0, 1, 2, 5}),
                ("2025-03-25", 504.0, 0.0007, {0, 1, 3, 5}),
                ("2025-03-26", 501.0, 0.0005, {0, 2, 4, 6}),
                ("2025-03-27", 507.0, 0.0008, {1, 2, 3, 5}),
            ]
        )

        vortex = VortexGenerator(provider).generate(
            VortexRequest(symbol="SPY", regime=Regime.CRAB, length_days=2, seed=19)
        )

        assert len(vortex.bars) == 780
        for bar in vortex.bars:
            low_price = float(bar["l"])
            open_price = float(bar["o"])
            close_price = float(bar["c"])
            high_price = float(bar["h"])
            vwap = float(bar["vw"])
            assert low_price <= min(open_price, close_price)
            assert max(open_price, close_price) <= high_price
            assert low_price <= vwap <= high_price
            assert int(bar["v"]) >= 1
            assert int(bar["n"]) >= 1

    def test_session_template_classifies_bull_bear_and_crab_regimes(self):
        provider = _FakeMarketDataProvider()
        generator = VortexGenerator(provider)

        bull_template = generator._session_to_template(
            _make_session("2025-03-24", 500.0, base_delta=0.0008, direction_pattern={0, 1, 2, 3, 5, 6})
        )
        bear_template = generator._session_to_template(
            _make_session("2025-03-25", 500.0, base_delta=0.0008, direction_pattern={1})
        )
        crab_template = generator._session_to_template(
            _make_session("2025-03-26", 500.0, base_delta=0.00008, direction_pattern={0}, pattern_modulus=2)
        )

        assert bull_template is not None and bull_template.regime_label == Regime.BULL
        assert bear_template is not None and bear_template.regime_label == Regime.BEAR
        assert crab_template is not None and crab_template.regime_label == Regime.CRAB

    def test_strong_bias_pushes_session_return_further_toward_requested_regime(self):
        provider = self._build_provider_with_reference_sessions(
            [
                ("2025-03-24", 500.0, 0.00045, {0, 2, 4}),
                ("2025-03-25", 503.0, 0.00045, {1, 3, 5}),
                ("2025-03-26", 501.0, 0.00045, {0, 1, 4}),
                ("2025-03-27", 502.0, 0.00045, {2, 3, 6}),
            ]
        )
        generator = VortexGenerator(provider)

        soft_vortex = generator.generate(
            VortexRequest(
                symbol="SPY",
                regime=Regime.BULL,
                length_days=1,
                seed=11,
                regime_bias=VortexRegimeBias.SOFT,
            )
        )
        strong_vortex = generator.generate(
            VortexRequest(
                symbol="SPY",
                regime=Regime.BULL,
                length_days=1,
                seed=11,
                regime_bias=VortexRegimeBias.STRONG,
            )
        )

        assert _session_log_return(strong_vortex.bars) > _session_log_return(soft_vortex.bars)

    def test_higher_volatility_templates_produce_higher_generated_return_dispersion(self):
        low_vol_provider = self._build_provider_with_reference_sessions(
            [
                ("2025-03-24", 500.0, 0.00015, {0, 1, 2, 5}),
                ("2025-03-25", 501.0, 0.00015, {0, 1, 2, 5}),
                ("2025-03-26", 502.0, 0.00015, {0, 1, 2, 5}),
                ("2025-03-27", 503.0, 0.00015, {0, 1, 2, 5}),
            ]
        )
        high_vol_provider = self._build_provider_with_reference_sessions(
            [
                ("2025-03-24", 500.0, 0.0012, {0, 1, 2, 5}),
                ("2025-03-25", 501.0, 0.0011, {0, 1, 3, 5}),
                ("2025-03-26", 502.0, 0.00125, {0, 2, 4, 6}),
                ("2025-03-27", 503.0, 0.00115, {1, 2, 3, 5}),
            ]
        )

        low_vol_vortex = VortexGenerator(low_vol_provider).generate(
            VortexRequest(symbol="SPY", regime=Regime.CRAB, length_days=1, seed=29)
        )
        high_vol_vortex = VortexGenerator(high_vol_provider).generate(
            VortexRequest(symbol="SPY", regime=Regime.CRAB, length_days=1, seed=29)
        )

        assert _minute_return_stdev(high_vol_vortex.bars) > (_minute_return_stdev(low_vol_vortex.bars) * 2.0)
