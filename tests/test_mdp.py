import tempfile
import unittest
from abc import ABC
from parabolic.mdp import MarketDataProvider, CachedMarketDataProvider

class FakeAlpacaMarketDataProvider(MarketDataProvider, ABC):

    def __init__(self):
        self.get_bars_calls = []
        self.get_latest_bar_calls = []

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        adjustment: str = "all",
        feed: str | None = None,
    ) -> list[dict]:
        self.get_bars_calls.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "start": start,
                "end": end,
                "adjustment": adjustment,
                "feed": feed,
            }
        )
        return [
            {
                "t": "2024-01-02T09:30:00Z",
                "o": 100.0,
                "h": 101.0,
                "l": 99.5,
                "c": 100.5,
                "v": 1000,
            }
        ]

    def get_latest_bar(self, symbol: str, feed: str | None = None) -> dict | None:
        self.get_latest_bar_calls.append({"symbol": symbol, "feed": feed})
        return {
            "t": "2024-01-02T16:00:00Z",
            "o": 100.0,
            "h": 102.0,
            "l": 99.0,
            "c": 101.5,
            "v": 2500,
        }

class TestMarketDataProvider(unittest.TestCase):

    def test_market_data_provider_is_abstract(self):
        with self.assertRaises(TypeError):
            MarketDataProvider()

    def test_cached_provider_downloads_then_reads_bars_from_cache(self):
        fake_provider = FakeAlpacaMarketDataProvider()

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = CachedMarketDataProvider(cache_dir=tmpdir, alpaca_provider=fake_provider)

            first_result = provider.get_bars(
                symbol="SPY",
                timeframe="1Day",
                start="2024-01-01T00:00:00Z",
                end="2024-01-31T00:00:00Z",
            )
            second_result = provider.get_bars(
                symbol="SPY",
                timeframe="1Day",
                start="2024-01-01T00:00:00Z",
                end="2024-01-31T00:00:00Z",
            )

            assert first_result == second_result
            assert len(fake_provider.get_bars_calls) == 1

    def test_cached_provider_downloads_then_reads_latest_bar_from_cache(self):
        fake_provider = FakeAlpacaMarketDataProvider()

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = CachedMarketDataProvider(cache_dir=tmpdir, alpaca_provider=fake_provider)

            first_result = provider.get_latest_bar(symbol="QQQ")
            second_result = provider.get_latest_bar(symbol="QQQ")

            assert first_result == second_result
            assert len(fake_provider.get_latest_bar_calls) == 1
