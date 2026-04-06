from __future__ import annotations
from copy import deepcopy
from typing import Any, Callable
from parabolic.mdp import MarketDataProvider

class TradingContext:

    def __init__(
        self,
        t: int,
        snapshot: list[dict[str, float]],
        asset_name: str = "TLT",
        **extras: Any,
    ):
        self.market = snapshot
        self.asset_name = asset_name
        self.t = t
        for key, value in extras.items():
            setattr(self, key, value)

class ContextOrchestrator:

    def __init__(
        self,
        market_data_provider: MarketDataProvider | None = None,
        snapshots: list[dict[str, float]] | None = None,
        asset_name: str = "TLT",
        start_date: str | None = None,
        end_date: str | None = None,
        timeframe: str = "1Day",
        adjustment: str = "all",
        feed: str | None = None,
        context_factory: Callable[[int, list[dict[str, float]], str, dict[str, Any]], dict[str, Any]] | None = None,
        extra_context: dict[str, Any] | None = None,
    ):
        self.market_data_provider = market_data_provider
        self.asset_name = asset_name
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.adjustment = adjustment
        self.feed = feed
        self.context_factory = context_factory
        self.extra_context = extra_context or {}
        self._snapshots = snapshots or []
        self.raw_bars: list[dict[str, Any]] = []
        self._loaded = bool(self._snapshots)

    def _normalize_bars(self, bars: list[dict[str, Any]]) -> list[dict[str, float]]:
        snapshots: list[dict[str, float]] = []
        for bar in bars:
            close_price = bar.get("c")
            if close_price is None:
                continue
            snapshots.append({self.asset_name: float(close_price)})
        return snapshots

    def _load_market_data(self) -> None:
        if self._loaded:
            return

        if self.market_data_provider is None:
            self._loaded = True
            return

        if self.start_date is None or self.end_date is None:
            self._loaded = True
            return

        self.raw_bars = self.market_data_provider.get_bars(
            symbol=self.asset_name,
            timeframe=self.timeframe,
            start=self.start_date,
            end=self.end_date,
            adjustment=self.adjustment,
            feed=self.feed,
        )
        self._snapshots = self._normalize_bars(self.raw_bars)
        self._loaded = True

    def get_snapshots(self) -> list[dict[str, float]]:
        self._load_market_data()
        return self._snapshots

    def build_context(self, t: int) -> TradingContext:
        snapshots = self.get_snapshots()
        context_payload: dict[str, Any] = {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "timeframe": self.timeframe,
            "adjustment": self.adjustment,
            "feed": self.feed,
            "bar": self.raw_bars[t] if t < len(self.raw_bars) else None,
            "bars": self.raw_bars[:t + 1] if self.raw_bars else [],
            "session_market": snapshots,
            "session_length": len(snapshots),
            "is_session_start": t == 0,
            "is_session_end": t == (len(snapshots) - 1),
        }
        context_payload.update(self.extra_context)
        if self.context_factory is not None:
            context_payload.update(
                self.context_factory(t, snapshots, self.asset_name, dict(context_payload))
            )
        return TradingContext(
            t=t,
            snapshot=snapshots[:t + 1],
            asset_name=self.asset_name,
            **context_payload,
        )

    def split_into_daily_orchestrators(self) -> list[tuple[str | None, "ContextOrchestrator"]]:
        snapshots = self.get_snapshots()
        if not snapshots:
            return []

        if self.timeframe == "1Day" or not self.raw_bars:
            sessions: list[tuple[str | None, ContextOrchestrator]] = []
            for index, snapshot in enumerate(snapshots):
                session_snapshots = [snapshot]
                if index > 0:
                    session_snapshots = [snapshots[index - 1], snapshot]

                session_orchestrator = ContextOrchestrator(
                    snapshots=session_snapshots,
                    asset_name=self.asset_name,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    timeframe=self.timeframe,
                    adjustment=self.adjustment,
                    feed=self.feed,
                    context_factory=self.context_factory,
                    extra_context=dict(self.extra_context),
                )
                session_raw_bars: list[dict[str, Any]] = []
                if index > 0 and (index - 1) < len(self.raw_bars):
                    session_raw_bars.append(self.raw_bars[index - 1])
                if index < len(self.raw_bars):
                    session_raw_bars.append(self.raw_bars[index])
                session_orchestrator.raw_bars = session_raw_bars
                session_orchestrator._loaded = True
                sessions.append((None, session_orchestrator))
            return sessions

        grouped: dict[str, dict[str, Any]] = {}
        ordered_dates: list[str] = []

        for index, bar in enumerate(self.raw_bars):
            timestamp = str(bar.get("t", ""))
            session_date = timestamp.split("T")[0] if "T" in timestamp else timestamp[:10]
            if session_date not in grouped:
                grouped[session_date] = {
                    "snapshots": [],
                    "raw_bars": [],
                    "history_snapshots": [],
                    "history_raw_bars": [],
                }
                ordered_dates.append(session_date)
                if index > 0 and (index - 1) < len(snapshots):
                    grouped[session_date]["history_snapshots"] = [snapshots[index - 1]]
                    grouped[session_date]["history_raw_bars"] = [self.raw_bars[index - 1]]
            grouped[session_date]["raw_bars"].append(bar)
            if index < len(snapshots):
                grouped[session_date]["snapshots"].append(snapshots[index])

        sessions = []
        for session_date in ordered_dates:
            payload = grouped[session_date]
            session_snapshots = payload["history_snapshots"] + payload["snapshots"]
            session_raw_bars = payload["history_raw_bars"] + payload["raw_bars"]
            session_orchestrator = ContextOrchestrator(
                snapshots=session_snapshots,
                asset_name=self.asset_name,
                start_date=session_date,
                end_date=session_date,
                timeframe=self.timeframe,
                adjustment=self.adjustment,
                feed=self.feed,
                context_factory=self.context_factory,
                extra_context=dict(self.extra_context),
            )
            session_orchestrator.raw_bars = session_raw_bars
            session_orchestrator._loaded = True
            sessions.append((session_date, session_orchestrator))
        return sessions