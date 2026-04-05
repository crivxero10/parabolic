from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import requests
from tinydb import Query, TinyDB

class MarketDataProvider(ABC):
    """Abstract interface for market data providers."""

    @abstractmethod
    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        adjustment: str = "all",
        feed: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return historical bars for a symbol."""

    @abstractmethod
    def get_latest_bar(self, symbol: str, feed: str | None = None) -> dict[str, Any] | None:
        """Return the latest bar for a symbol."""


class AlpacaMarketDataProvider(MarketDataProvider):
    """Market data provider backed by the Alpaca Market Data API."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://data.alpaca.markets",
        session: requests.Session | None = None,
        timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret,
            }
        )

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        adjustment: str = "all",
        feed: str | None = None,
    ) -> list[dict[str, Any]]:
        url = f"{self.base_url}/v2/stocks/bars"
        params: dict[str, Any] = {
            "symbols": symbol,
            "timeframe": timeframe,
            "start": start,
            "end": end,
            "adjustment": adjustment,
        }
        if feed is not None:
            params["feed"] = feed

        bars: list[dict[str, Any]] = []
        page_token: str | None = None

        while True:
            request_params = dict(params)
            if page_token is not None:
                request_params["page_token"] = page_token

            response = self.session.get(url, params=request_params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()

            bars.extend(payload.get("bars", {}).get(symbol, []))
            page_token = payload.get("next_page_token")
            if not page_token:
                break

        return bars

    def get_latest_bar(self, symbol: str, feed: str | None = None) -> dict[str, Any] | None:
        url = f"{self.base_url}/v2/stocks/bars/latest"
        params: dict[str, Any] = {"symbols": symbol}
        if feed is not None:
            params["feed"] = feed

        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        return payload.get("bars", {}).get(symbol)


class CachedMarketDataProvider(MarketDataProvider):
    """TinyDB-backed cache for Alpaca market data.

    Historical and latest bar reads always check cache first. On a miss, data is
    downloaded from Alpaca, cached locally, and then returned.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        alpaca_provider: AlpacaMarketDataProvider,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db = TinyDB(self.cache_dir / "alpaca_market_data.json")
        self.alpaca_provider = alpaca_provider
        self.query = Query()

    def _find_one(self, key: dict[str, Any]) -> dict[str, Any] | None:
        return self.db.get(
            (self.query.kind == key["kind"])
            & (self.query.symbol == key["symbol"])
            & (self.query.timeframe == key["timeframe"])
            & (self.query.start == key["start"])
            & (self.query.end == key["end"])
            & (self.query.adjustment == key["adjustment"])
            & (self.query.feed == key["feed"])
        )

    def _upsert_one(self, key: dict[str, Any], data: Any) -> None:
        document = {**key, "data": data}
        self.db.upsert(
            document,
            (self.query.kind == key["kind"])
            & (self.query.symbol == key["symbol"])
            & (self.query.timeframe == key["timeframe"])
            & (self.query.start == key["start"])
            & (self.query.end == key["end"])
            & (self.query.adjustment == key["adjustment"])
            & (self.query.feed == key["feed"]),
        )

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        adjustment: str = "all",
        feed: str | None = None,
    ) -> list[dict[str, Any]]:
        key = {
            "kind": "bars",
            "symbol": symbol,
            "timeframe": timeframe,
            "start": start,
            "end": end,
            "adjustment": adjustment,
            "feed": feed,
        }
        cached = self._find_one(key)
        if cached is not None:
            return cached["data"]

        bars = self.alpaca_provider.get_bars(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            adjustment=adjustment,
            feed=feed,
        )
        self._upsert_one(key, bars)
        return bars

    def get_latest_bar(self, symbol: str, feed: str | None = None) -> dict[str, Any] | None:
        key = {
            "kind": "latest_bar",
            "symbol": symbol,
            "timeframe": None,
            "start": None,
            "end": None,
            "adjustment": None,
            "feed": feed,
        }
        cached = self._find_one(key)
        if cached is not None:
            return cached["data"]

        latest_bar = self.alpaca_provider.get_latest_bar(symbol=symbol, feed=feed)
        self._upsert_one(key, latest_bar)
        return latest_bar