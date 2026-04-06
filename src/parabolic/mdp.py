from __future__ import annotations

import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import date as date_cls, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests


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
        max_parallelism: int = 32,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_parallelism = max_parallelism
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret,
            }
        )

    def _parse_session_day(self, value: str) -> date_cls:
        return date_cls.fromisoformat(value[:10])

    def _to_utc_z(self, dt: datetime) -> str:
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def _iter_calendar_days(self, start: str, end: str) -> list[str]:
        start_day = self._parse_session_day(start)
        end_day = self._parse_session_day(end)
        days: list[str] = []
        current_day = start_day
        while current_day <= end_day:
            if current_day.weekday() < 5:
                days.append(current_day.isoformat())
            current_day += timedelta(days=1)
        return days

    def get_regular_session_1m_bars(
        self,
        day: str | date_cls,
        symbol: str,
        adjustment: str = "all",
        feed: str | None = None,
    ) -> list[dict[str, Any]]:
        market_tz = ZoneInfo("America/New_York")
        session_day = self._parse_session_day(day) if isinstance(day, str) else day

        start_ny = datetime.combine(session_day, time(9, 30), tzinfo=market_tz)
        end_ny = datetime.combine(session_day, time(16, 0), tzinfo=market_tz)

        params: dict[str, Any] = {
            "symbols": symbol,
            "timeframe": "1Min",
            "start": self._to_utc_z(start_ny),
            "end": self._to_utc_z(end_ny),
            "adjustment": adjustment,
            "limit": 1000,
        }
        if feed is not None:
            params["feed"] = feed

        response = self.session.get(
            f"{self.base_url}/v2/stocks/bars",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        bars = payload.get("bars", {}).get(symbol, [])

        if len(bars) == 390:
            return bars

        def parse_bar_time_utc(ts: str) -> datetime:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))

        bars_by_minute_ny: dict[datetime, dict[str, Any]] = {}
        for bar in bars:
            bar_utc = parse_bar_time_utc(bar["t"])
            bar_ny = bar_utc.astimezone(market_tz).replace(second=0, microsecond=0)
            bars_by_minute_ny[bar_ny] = bar

        expected_minutes_ny = [start_ny + timedelta(minutes=i) for i in range(390)]
        normalized: list[dict[str, Any]] = []
        prev_bar: dict[str, Any] | None = None

        first_real_bar = None
        for minute_ny in expected_minutes_ny:
            if minute_ny in bars_by_minute_ny:
                first_real_bar = bars_by_minute_ny[minute_ny]
                break

        for minute_ny in expected_minutes_ny:
            bar = bars_by_minute_ny.get(minute_ny)
            if bar is not None:
                current = deepcopy(bar)
            else:
                seed = prev_bar if prev_bar is not None else first_real_bar
                if seed is None:
                    return []
                current = deepcopy(seed)
                current["t"] = self._to_utc_z(minute_ny)

            normalized.append(current)
            prev_bar = current

        return normalized

    async def _get_regular_session_1m_bars_async(
        self,
        semaphore: asyncio.Semaphore,
        day: str,
        symbol: str,
        adjustment: str,
        feed: str | None,
    ) -> list[dict[str, Any]]:
        async with semaphore:
            return await asyncio.to_thread(
                self.get_regular_session_1m_bars,
                day,
                symbol,
                adjustment,
                feed,
            )

    async def _get_regular_session_1m_bars_range_async(
        self,
        symbol: str,
        start: str,
        end: str,
        adjustment: str = "all",
        feed: str | None = None,
    ) -> list[dict[str, Any]]:
        calendar_days = self._iter_calendar_days(start, end)
        if not calendar_days:
            return []

        semaphore = asyncio.Semaphore(min(self.max_parallelism, len(calendar_days)))
        tasks = [
            self._get_regular_session_1m_bars_async(
                semaphore=semaphore,
                day=calendar_day,
                symbol=symbol,
                adjustment=adjustment,
                feed=feed,
            )
            for calendar_day in calendar_days
        ]
        per_day_bars = await asyncio.gather(*tasks)

        bars: list[dict[str, Any]] = []
        for day_bars in per_day_bars:
            bars.extend(day_bars)
        return bars

    def get_regular_session_1m_bars_range(
        self,
        symbol: str,
        start: str,
        end: str,
        adjustment: str = "all",
        feed: str | None = None,
    ) -> list[dict[str, Any]]:
        return asyncio.run(
            self._get_regular_session_1m_bars_range_async(
                symbol=symbol,
                start=start,
                end=end,
                adjustment=adjustment,
                feed=feed,
            )
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
        if timeframe == "1Min":
            return self.get_regular_session_1m_bars_range(
                symbol=symbol,
                start=start,
                end=end,
                adjustment=adjustment,
                feed=feed,
            )

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
    """JSON-file-backed cache for Alpaca market data."""

    def __init__(
        self,
        cache_dir: str | Path,
        alpaca_provider: AlpacaMarketDataProvider,
        refresh: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.alpaca_provider = alpaca_provider
        self.refresh = refresh

    def _slug(self, value: str | None) -> str:
        if value is None or value == "":
            return "none"
        return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)

    def _hash_key(self, payload: dict[str, Any]) -> str:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def _range_cache_path(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        adjustment: str,
        feed: str | None,
    ) -> Path:
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start": start,
            "end": end,
            "adjustment": adjustment,
            "feed": feed,
        }
        filename = (
            f"bars_{self._slug(symbol)}_{self._slug(timeframe)}_"
            f"{self._hash_key(payload)}.json"
        )
        return self.cache_dir / filename

    def _day_cache_path(
        self,
        symbol: str,
        timeframe: str,
        day: str,
        adjustment: str,
        feed: str | None,
    ) -> Path:
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "day": day,
            "adjustment": adjustment,
            "feed": feed,
        }
        filename = (
            f"bars_day_{self._slug(symbol)}_{self._slug(timeframe)}_{self._slug(day)}_"
            f"{self._hash_key(payload)}.json"
        )
        return self.cache_dir / filename

    def _latest_cache_path(self, symbol: str, feed: str | None) -> Path:
        payload = {"symbol": symbol, "feed": feed}
        filename = f"latest_bar_{self._slug(symbol)}_{self._hash_key(payload)}.json"
        return self.cache_dir / filename

    def _read_json_file(self, path: Path) -> Any | None:
        if self.refresh or not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write_json_file(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.refresh and path.exists():
            path.unlink()
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

    async def _get_or_fetch_day_async(
        self,
        semaphore: asyncio.Semaphore,
        symbol: str,
        timeframe: str,
        day: str,
        adjustment: str,
        feed: str | None,
    ) -> list[dict[str, Any]]:
        path = self._day_cache_path(
            symbol=symbol,
            timeframe=timeframe,
            day=day,
            adjustment=adjustment,
            feed=feed,
        )
        cached = self._read_json_file(path)
        if cached is not None:
            return cached

        async with semaphore:
            bars = await asyncio.to_thread(
                self.alpaca_provider.get_regular_session_1m_bars,
                day,
                symbol,
                adjustment,
                feed,
            )
        self._write_json_file(path, bars)
        return bars

    async def _get_bars_1m_async(
        self,
        symbol: str,
        start: str,
        end: str,
        adjustment: str,
        feed: str | None,
    ) -> list[dict[str, Any]]:
        calendar_days = self.alpaca_provider._iter_calendar_days(start, end)
        if not calendar_days:
            return []

        semaphore = asyncio.Semaphore(min(self.alpaca_provider.max_parallelism, len(calendar_days)))
        tasks = [
            self._get_or_fetch_day_async(
                semaphore=semaphore,
                symbol=symbol,
                timeframe="1Min",
                day=day,
                adjustment=adjustment,
                feed=feed,
            )
            for day in calendar_days
        ]
        per_day_bars = await asyncio.gather(*tasks)

        bars: list[dict[str, Any]] = []
        for day_bars in per_day_bars:
            bars.extend(day_bars)
        return bars

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        adjustment: str = "all",
        feed: str | None = None,
    ) -> list[dict[str, Any]]:
        if timeframe == "1Min":
            return asyncio.run(
                self._get_bars_1m_async(
                    symbol=symbol,
                    start=start,
                    end=end,
                    adjustment=adjustment,
                    feed=feed,
                )
            )

        path = self._range_cache_path(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            adjustment=adjustment,
            feed=feed,
        )
        cached = self._read_json_file(path)
        if cached is not None:
            return cached

        bars = self.alpaca_provider.get_bars(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            adjustment=adjustment,
            feed=feed,
        )
        self._write_json_file(path, bars)
        return bars

    def get_latest_bar(self, symbol: str, feed: str | None = None) -> dict[str, Any] | None:
        path = self._latest_cache_path(symbol=symbol, feed=feed)
        cached = self._read_json_file(path)
        if cached is not None:
            return cached

        latest_bar = self.alpaca_provider.get_latest_bar(symbol=symbol, feed=feed)
        self._write_json_file(path, latest_bar)
        return latest_bar