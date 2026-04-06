from __future__ import annotations
from copy import deepcopy
from datetime import datetime, time, timedelta
from typing import Any, Callable
from zoneinfo import ZoneInfo
from parabolic.brokerage import Brokerage
from parabolic.mdp import MarketDataProvider
from parabolic.orchestrator import TradingContext, ContextOrchestrator

class SimulationStep:

    def __init__(self, t: int, realized_pnl: float, unrealized_pnl: float):
        self.t = t
        self.realized_pnl = realized_pnl
        self.unrealized_pnl = unrealized_pnl

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    def __repr__(self) -> str:
        return (
            f"SimulationStep(t={self.t}, realized_pnl={self.realized_pnl}, "
            f"unrealized_pnl={self.unrealized_pnl}, total_pnl={self.total_pnl})"
        )


class DailySimulationResult:

    def __init__(
        self,
        session_date: str | None,
        steps: list[SimulationStep],
        end_balance: float,
        end_available_cash: float,
    ):
        self.session_date = session_date
        self.steps = steps
        self.end_balance = end_balance
        self.end_available_cash = end_available_cash

    def __repr__(self) -> str:
        return (
            f"DailySimulationResult(session_date={self.session_date}, "
            f"end_balance={self.end_balance}, end_available_cash={self.end_available_cash}, "
            f"steps={self.steps})"
        )


class Backtester:
    def __init__(
        self,
        snapshots: list[dict[str, float]] | None = None,
        strategy: Callable[[TradingContext], None] | None = None,
        brokerage: Brokerage | None = None,
        asset_name: str = "TLT",
        market_data_provider: MarketDataProvider | None = None,
        context_orchestrator: ContextOrchestrator | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        timeframe: str = "1Day",
        adjustment: str = "all",
        feed: str | None = None,
    ):
        self.strategy = strategy
        self.brokerage = brokerage
        self.asset_name = asset_name
        self.context_orchestrator = context_orchestrator or ContextOrchestrator(
            market_data_provider=market_data_provider,
            snapshots=snapshots,
            asset_name=asset_name,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            adjustment=adjustment,
            feed=feed,
        )
        self.snapshots = self.context_orchestrator.get_snapshots()
        self.simulation_steps: list[SimulationStep] = []
        self._iter_brokerage: Brokerage | None = None
        self._iter_strategy: Callable[[TradingContext], None] | None = None

    def _build_stale_step(self) -> SimulationStep:
        return SimulationStep(
            t=1,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
        )

    def __iter__(self):
        if self.simulation_steps:
            return iter(self.simulation_steps)

        brokerage = self._iter_brokerage or self.brokerage
        strategy = self._iter_strategy or self.strategy
        if brokerage is None or strategy is None:
            raise ValueError(
                "Backtester iterator is not configured. Pass a brokerage and strategy at construction time, call simulate(...)/iter_simulation(...), or set iteration inputs before iterating."
            )
        return self.iter_simulation(brokerage, strategy)

    def _resolve_strategy(
        self,
        strategy: Callable[[TradingContext], None] | None,
    ) -> Callable[[TradingContext], None]:
        resolved_strategy = strategy or self.strategy
        if resolved_strategy is None:
            raise ValueError(
                "No strategy configured. Pass a strategy at construction time or to the method call."
            )
        return resolved_strategy

    def _resolve_brokerage(self, brokerage: Brokerage | None) -> Brokerage:
        resolved_brokerage = brokerage or self.brokerage
        if resolved_brokerage is None:
            raise ValueError(
                "No brokerage configured. Pass a brokerage at construction time or to the method call."
            )
        return resolved_brokerage

    def _parse_bar_timestamp_ny(self, ts: str) -> datetime:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(ZoneInfo("America/New_York"))

    def _is_regular_session_bar(self, bar: dict[str, Any]) -> bool:
        timestamp = bar.get("t")
        if not timestamp:
            return False
        dt_ny = self._parse_bar_timestamp_ny(str(timestamp))
        if dt_ny.weekday() >= 5:
            return False
        return time(9, 30) <= dt_ny.time() < time(16, 0)

    def _normalize_intraday_session(
        self,
        session_date: str,
        raw_bars: list[dict[str, Any]],
        snapshots: list[dict[str, float]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
        market_tz = ZoneInfo("America/New_York")
        start_ny = datetime.fromisoformat(f"{session_date}T09:30:00").replace(tzinfo=market_tz)
        expected_minutes = [start_ny + timedelta(minutes=i) for i in range(390)]

        bars_by_minute: dict[datetime, dict[str, Any]] = {}
        snapshots_by_minute: dict[datetime, dict[str, float]] = {}

        for bar, snapshot in zip(raw_bars, snapshots):
            timestamp = bar.get("t")
            if not timestamp:
                continue
            dt_ny = self._parse_bar_timestamp_ny(str(timestamp)).replace(second=0, microsecond=0)
            if not (time(9, 30) <= dt_ny.time() < time(16, 0)):
                continue
            bars_by_minute[dt_ny] = deepcopy(bar)
            snapshots_by_minute[dt_ny] = deepcopy(snapshot)

        first_real_bar = None
        first_real_snapshot = None
        for minute_ny in expected_minutes:
            if minute_ny in bars_by_minute and minute_ny in snapshots_by_minute:
                first_real_bar = deepcopy(bars_by_minute[minute_ny])
                first_real_snapshot = deepcopy(snapshots_by_minute[minute_ny])
                break

        normalized_bars: list[dict[str, Any]] = []
        normalized_snapshots: list[dict[str, float]] = []
        prev_bar: dict[str, Any] | None = None
        prev_snapshot: dict[str, float] | None = None

        for minute_ny in expected_minutes:
            if minute_ny in bars_by_minute and minute_ny in snapshots_by_minute:
                current_bar = deepcopy(bars_by_minute[minute_ny])
                current_snapshot = deepcopy(snapshots_by_minute[minute_ny])
            else:
                if prev_bar is not None and prev_snapshot is not None:
                    current_bar = deepcopy(prev_bar)
                    current_snapshot = deepcopy(prev_snapshot)
                elif first_real_bar is not None and first_real_snapshot is not None:
                    current_bar = deepcopy(first_real_bar)
                    current_snapshot = deepcopy(first_real_snapshot)
                else:
                    continue
                current_bar["t"] = minute_ny.astimezone(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")

            normalized_bars.append(current_bar)
            normalized_snapshots.append(current_snapshot)
            prev_bar = current_bar
            prev_snapshot = current_snapshot

        return normalized_bars, normalized_snapshots

    def _build_daily_session_orchestrators(self) -> list[tuple[str | None, ContextOrchestrator]]:
        if self.context_orchestrator.timeframe != "1Min" or not getattr(self.context_orchestrator, "raw_bars", None):
            return self.context_orchestrator.split_into_daily_orchestrators()

        grouped: dict[str, dict[str, list[Any]]] = {}
        ordered_dates: list[str] = []

        raw_bars = getattr(self.context_orchestrator, "raw_bars", [])
        snapshots = self.context_orchestrator.get_snapshots()

        for bar, snapshot in zip(raw_bars, snapshots):
            if not self._is_regular_session_bar(bar):
                continue
            dt_ny = self._parse_bar_timestamp_ny(str(bar["t"])).replace(second=0, microsecond=0)
            session_date = dt_ny.date().isoformat()
            if session_date not in grouped:
                grouped[session_date] = {"raw_bars": [], "snapshots": []}
                ordered_dates.append(session_date)
            grouped[session_date]["raw_bars"].append(bar)
            grouped[session_date]["snapshots"].append(snapshot)

        sessions: list[tuple[str | None, ContextOrchestrator]] = []
        for session_date in ordered_dates:
            payload = grouped[session_date]
            normalized_bars, normalized_snapshots = self._normalize_intraday_session(
                session_date=session_date,
                raw_bars=payload["raw_bars"],
                snapshots=payload["snapshots"],
            )
            if not normalized_snapshots:
                continue
            session_orchestrator = ContextOrchestrator(
                snapshots=normalized_snapshots,
                asset_name=self.asset_name,
                start_date=session_date,
                end_date=session_date,
                timeframe=self.context_orchestrator.timeframe,
                adjustment=self.context_orchestrator.adjustment,
                feed=self.context_orchestrator.feed,
                context_factory=self.context_orchestrator.context_factory,
                extra_context=dict(self.context_orchestrator.extra_context),
            )
            session_orchestrator.raw_bars = normalized_bars
            session_orchestrator._loaded = True
            sessions.append((session_date, session_orchestrator))

        return sessions

    def iter_simulation(
        self,
        brokerage: Brokerage | None = None,
        strategy: Callable[[TradingContext], None] | None = None,
    ):
        brokerage = self._resolve_brokerage(brokerage)
        strategy = self._resolve_strategy(strategy)
        self._iter_brokerage = brokerage
        self._iter_strategy = strategy
        self.snapshots = self.context_orchestrator.get_snapshots()
        self.simulation_steps = []

        if not self.snapshots:
            return

        stale_step = self._build_stale_step()
        self.simulation_steps.append(stale_step)
        yield stale_step

        for t in range(1, len(self.snapshots)):
            step = self._simulate_step(
                t=t,
                brokerage=brokerage,
                strategy=strategy,
            )
            step.t = t + 1
            self.simulation_steps.append(step)
            yield step

    def _build_context(self, t: int, brokerage: Brokerage) -> TradingContext:
        ctx = self.context_orchestrator.build_context(t)
        ctx.brokerage = brokerage
        return ctx

    def _simulate_step(
        self,
        t: int,
        brokerage: Brokerage,
        strategy: Callable[[TradingContext], None],
    ) -> SimulationStep:
        ctx = self._build_context(t, brokerage)
        brokerage.execute_all_deferred(ctx)
        strategy(ctx)
        market_snapshot = self.snapshots[t]
        realized_pnl = brokerage.get_total_realized_pnl(market_snapshot)
        unrealized_pnl = brokerage.get_total_unrealized_pnl(market_snapshot)
        return SimulationStep(
            t=t,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
        )

    def _simulate_single_snapshot_session(
        self,
        brokerage: Brokerage,
        strategy: Callable[[TradingContext], None],
    ) -> list[SimulationStep]:
        self.simulation_steps = []

        if not self.snapshots:
            return self.simulation_steps

        ctx = self._build_context(0, brokerage)
        brokerage.execute_all_deferred(ctx)
        strategy(ctx)
        market_snapshot = self.snapshots[0]
        realized_pnl = brokerage.get_total_realized_pnl(market_snapshot)
        unrealized_pnl = brokerage.get_total_unrealized_pnl(market_snapshot)
        self.simulation_steps.append(
            SimulationStep(
                t=1,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
            )
        )
        return self.simulation_steps

    def simulate(
        self,
        brokerage: Brokerage | None = None,
        strategy: Callable[[TradingContext], None] | None = None,
    ) -> list[SimulationStep]:
        for _ in self.iter_simulation(brokerage, strategy):
            pass
        return self.simulation_steps

    def get_rolling_pnl(
        self,
        brokerage: Brokerage | None = None,
        strategy: Callable[[TradingContext], None] | None = None,
    ) -> list[float]:
        self.simulate(brokerage, strategy)
        return [step.total_pnl for step in self.simulation_steps]

    def get_rolling_pnl_comparison(
        self,
        brokerageA: Brokerage,
        brokerageB: Brokerage,
        strategyA: Callable[[TradingContext], None] | None = None,
        strategyB: Callable[[TradingContext], None] | None = None,
    ) -> list[float]:
        pnlA = self.get_rolling_pnl(brokerageA, strategyA)
        pnlB = self.get_rolling_pnl(brokerageB, strategyB)
        return [round(b - a, 2) for a, b in zip(pnlA, pnlB)]

    def simulate_by_day(
        self,
        brokerage: Brokerage | None = None,
        strategy: Callable[[TradingContext], None] | None = None,
        brokerage_factory: Callable[[], Brokerage] | None = None,
        carry_state: bool = True,
    ) -> list[DailySimulationResult]:
        strategy = self._resolve_strategy(strategy)
        base_brokerage = self._resolve_brokerage(brokerage)
        session_results: list[DailySimulationResult] = []

        for session_date, session_orchestrator in self._build_daily_session_orchestrators():
            if carry_state:
                session_brokerage = base_brokerage
            else:
                session_brokerage = brokerage_factory() if brokerage_factory is not None else deepcopy(base_brokerage)
            session_backtester = Backtester(
                snapshots=session_orchestrator.get_snapshots(),
                strategy=strategy,
                brokerage=session_brokerage,
                asset_name=self.asset_name,
                context_orchestrator=session_orchestrator,
            )
            if len(session_backtester.snapshots) == 1:
                steps = session_backtester._simulate_single_snapshot_session(session_brokerage, strategy)
            else:
                steps = session_backtester.simulate(session_brokerage, strategy)

            final_market_snapshot = session_backtester.snapshots[-1] if session_backtester.snapshots else {}
            if any(units > 0 for units in session_brokerage.positions.values()):
                session_brokerage.liquidate(final_market_snapshot)

            session_results.append(
                DailySimulationResult(
                    session_date=session_date,
                    steps=steps,
                    end_balance=session_brokerage.balance,
                    end_available_cash=getattr(session_brokerage, "available_cash", session_brokerage.balance),
                )
            )

        return session_results

    def get_daily_balances(
        self,
        brokerage: Brokerage | None = None,
        strategy: Callable[[TradingContext], None] | None = None,
        brokerage_factory: Callable[[], Brokerage] | None = None,
        carry_state: bool = True,
    ) -> list[float]:
        return [
            result.end_balance
            for result in self.simulate_by_day(
                brokerage=brokerage,
                strategy=strategy,
                brokerage_factory=brokerage_factory,
                carry_state=carry_state,
            )
        ]


Engine = Backtester