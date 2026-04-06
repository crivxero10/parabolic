from __future__ import annotations
from copy import deepcopy
from datetime import datetime, time, timedelta
from typing import Any, Callable
from zoneinfo import ZoneInfo
from parabolic.brokerage import Brokerage
from parabolic.mdp import MarketDataProvider
from parabolic.orchestrator import TradingContext, ContextOrchestrator

class SimulationStep:

    def __init__(
        self,
        t: int,
        realized_pnl: float,
        unrealized_pnl: float,
        timestamp: str | None = None,
        balance: float | None = None,
        cash: float | None = None,
        position_value: float | None = None,
        equity: float | None = None,
        closed_trades: list[dict[str, Any]] | None = None,
    ):
        self.t = t
        self.realized_pnl = realized_pnl
        self.unrealized_pnl = unrealized_pnl
        self.timestamp = timestamp
        self.balance = balance
        self.cash = cash
        self.position_value = position_value
        self.equity = equity
        self.closed_trades = closed_trades or []

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    def __repr__(self) -> str:
        return (
            f"SimulationStep(t={self.t}, timestamp={self.timestamp}, realized_pnl={self.realized_pnl}, "
            f"unrealized_pnl={self.unrealized_pnl}, total_pnl={self.total_pnl}, balance={self.balance}, "
            f"cash={self.cash}, position_value={self.position_value}, equity={self.equity}, "
            f"closed_trades={self.closed_trades})"
        )


class DailySimulationResult:

    def __init__(
        self,
        session_date: str | None,
        steps: list[SimulationStep],
        end_balance: float,
        end_available_cash: float,
        daily_pnl_amount: float | None = None,
        daily_pnl_pct: float | None = None,
    ):
        self.session_date = session_date
        self.steps = steps
        self.end_balance = end_balance
        self.end_available_cash = end_available_cash
        self.daily_pnl_amount = daily_pnl_amount
        self.daily_pnl_pct = daily_pnl_pct

    def __repr__(self) -> str:
        return (
            f"DailySimulationResult(session_date={self.session_date}, "
            f"end_balance={self.end_balance}, end_available_cash={self.end_available_cash}, "
            f"daily_pnl_amount={self.daily_pnl_amount}, daily_pnl_pct={self.daily_pnl_pct}, "
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
        collect_equity_curve: bool = True,
        collect_closed_trades: bool = True,
        collect_daily_snapshots: bool = True,
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
        self.equity_curve: list[dict[str, Any]] = []
        self.closed_trades: list[dict[str, Any]] = []
        self.daily_snapshots: list[dict[str, Any]] = []
        self._starting_equity: float | None = None
        self.collect_equity_curve = collect_equity_curve
        self.collect_closed_trades = collect_closed_trades
        self.collect_daily_snapshots = collect_daily_snapshots

    def _build_stale_step(self) -> SimulationStep:
        return SimulationStep(
            t=1,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
        )

    def _extract_timestamp(self, t: int) -> str | None:
        raw_bars = getattr(self.context_orchestrator, "raw_bars", None)
        if raw_bars and 0 <= t < len(raw_bars):
            timestamp = raw_bars[t].get("t")
            return None if timestamp is None else str(timestamp)
        return None

    def _compute_position_value(self, market_snapshot: dict[str, float], brokerage: Brokerage) -> float:
        position_value = 0.0
        for asset_name, units in brokerage.positions.items():
            if units <= 0:
                continue
            if asset_name not in market_snapshot:
                continue
            position_value += float(units) * float(market_snapshot[asset_name])
        return position_value

    def _compute_total_equity(self, market_snapshot: dict[str, float], brokerage: Brokerage) -> float:
        cash = float(getattr(brokerage, "available_cash", brokerage.balance))
        position_value = self._compute_position_value(market_snapshot, brokerage)
        return cash + position_value

    def _record_equity_snapshot(
        self,
        *,
        t: int,
        brokerage: Brokerage,
        market_snapshot: dict[str, float],
    ) -> dict[str, Any]:
        timestamp = self._extract_timestamp(t)
        balance = float(brokerage.balance)
        cash = float(getattr(brokerage, "available_cash", brokerage.balance))
        position_value = self._compute_position_value(market_snapshot, brokerage)
        equity = cash + position_value
        snapshot = {
            "t": t,
            "timestamp": timestamp,
            "balance": balance,
            "cash": cash,
            "position_value": position_value,
            "equity": equity,
        }
        if self.collect_equity_curve or self.collect_daily_snapshots:
            self.equity_curve.append(snapshot)
        return snapshot

    def _record_closed_trade(self, trade: dict[str, Any]) -> None:
        if self.collect_closed_trades:
            self.closed_trades.append(dict(trade))

    def _collect_new_closed_trades(
        self,
        brokerage: Brokerage,
        closed_trade_cursor: int,
    ) -> tuple[list[dict[str, Any]], int]:
        if not self.collect_closed_trades:
            return [], closed_trade_cursor
        if not hasattr(brokerage, "get_closed_trades"):
            return [], closed_trade_cursor

        raw_closed_trades = getattr(brokerage, "_closed_trades", None)
        if isinstance(raw_closed_trades, list):
            all_closed_trades = raw_closed_trades
        else:
            all_closed_trades = brokerage.get_closed_trades()
            if not isinstance(all_closed_trades, list):
                return [], closed_trade_cursor

        if closed_trade_cursor < 0:
            closed_trade_cursor = 0
        if closed_trade_cursor > len(all_closed_trades):
            closed_trade_cursor = len(all_closed_trades)

        new_trades = all_closed_trades[closed_trade_cursor:]
        next_cursor = len(all_closed_trades)

        collected: list[dict[str, Any]] = []
        for trade in new_trades:
            normalized_trade = {
                "position_id": trade.get("position_id"),
                "asset": trade.get("asset"),
                "side": trade.get("side"),
                "entry_timestamp": trade.get("entry_timestamp"),
                "exit_timestamp": trade.get("exit_timestamp"),
                "entry_price": trade.get("entry_price"),
                "exit_price": trade.get("exit_price"),
                "quantity": trade.get("quantity"),
                "pnl_amount": trade.get("pnl_amount"),
                "pnl_pct": trade.get("pnl_pct"),
                "bars_held": trade.get("bars_held"),
                "fees": trade.get("fees"),
                "slippage": trade.get("slippage"),
            }
            self._record_closed_trade(normalized_trade)
            collected.append(normalized_trade)
        return collected, next_cursor

    def _build_daily_snapshots(self) -> list[dict[str, Any]]:
        if not self.collect_daily_snapshots:
            self.daily_snapshots = []
            return []
        if not self.equity_curve:
            return []

        daily_groups: dict[str, list[dict[str, Any]]] = {}
        for snapshot in self.equity_curve:
            timestamp = snapshot.get("timestamp")
            if not timestamp:
                continue
            date_key = str(timestamp)[:10]
            daily_groups.setdefault(date_key, []).append(snapshot)

        ordered_dates = sorted(daily_groups.keys())
        daily_snapshots: list[dict[str, Any]] = []
        previous_equity: float | None = None
        for date_key in ordered_dates:
            day_snapshots = daily_groups[date_key]
            ending_snapshot = day_snapshots[-1]
            ending_equity = float(ending_snapshot["equity"])
            if previous_equity is None:
                base_equity = (
                    self._starting_equity
                    if self._starting_equity is not None
                    else float(ending_snapshot["equity"])
                )
                daily_pnl_amount = ending_equity - base_equity
            else:
                daily_pnl_amount = ending_equity - previous_equity
                base_equity = previous_equity
            daily_pnl_pct = 0.0 if base_equity == 0 else daily_pnl_amount / base_equity
            daily_snapshots.append(
                {
                    "date": date_key,
                    "ending_balance": float(ending_snapshot["balance"]),
                    "ending_cash": float(ending_snapshot["cash"]),
                    "ending_position_value": float(ending_snapshot["position_value"]),
                    "ending_equity": ending_equity,
                    "daily_pnl_amount": daily_pnl_amount,
                    "daily_pnl_pct": daily_pnl_pct,
                }
            )
            previous_equity = ending_equity
        self.daily_snapshots = daily_snapshots
        return daily_snapshots

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
        self.equity_curve = []
        self.closed_trades = []
        self.daily_snapshots = []
        self._starting_equity = None

        if not self.snapshots:
            return

        self._starting_equity = self._compute_total_equity(self.snapshots[0], brokerage)
        closed_trade_cursor = 0
        stale_step = self._build_stale_step()
        self.simulation_steps.append(stale_step)
        yield stale_step

        for t in range(1, len(self.snapshots)):
            step, closed_trade_cursor = self._simulate_step(
                t=t,
                brokerage=brokerage,
                strategy=strategy,
                closed_trade_cursor=closed_trade_cursor,
            )
            step.t = t + 1
            self.simulation_steps.append(step)
            yield step

        self._build_daily_snapshots()

    def _build_context(self, t: int, brokerage: Brokerage) -> TradingContext:
        ctx = self.context_orchestrator.build_context(t)
        ctx.brokerage = brokerage
        return ctx

    def _simulate_step(
        self,
        t: int,
        brokerage: Brokerage,
        strategy: Callable[[TradingContext], None],
        closed_trade_cursor: int,
    ) -> tuple[SimulationStep, int]:
        ctx = self._build_context(t, brokerage)
        brokerage.execute_all_deferred(ctx)
        strategy(ctx)
        market_snapshot = self.snapshots[t]
        realized_pnl = brokerage.get_total_realized_pnl(market_snapshot)
        unrealized_pnl = brokerage.get_total_unrealized_pnl(market_snapshot)
        equity_snapshot = self._record_equity_snapshot(
            t=t,
            brokerage=brokerage,
            market_snapshot=market_snapshot,
        )
        closed_trades, next_closed_trade_cursor = self._collect_new_closed_trades(
            brokerage,
            closed_trade_cursor,
        )
        return (
            SimulationStep(
                t=t,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
                timestamp=equity_snapshot["timestamp"],
                balance=equity_snapshot["balance"],
                cash=equity_snapshot["cash"],
                position_value=equity_snapshot["position_value"],
                equity=equity_snapshot["equity"],
                closed_trades=closed_trades,
            ),
            next_closed_trade_cursor,
        )

    def _simulate_single_snapshot_session(
        self,
        brokerage: Brokerage,
        strategy: Callable[[TradingContext], None],
    ) -> list[SimulationStep]:
        self.simulation_steps = []
        self.equity_curve = []
        self.closed_trades = []
        self.daily_snapshots = []
        self._starting_equity = None

        if not self.snapshots:
            return self.simulation_steps

        self._starting_equity = self._compute_total_equity(self.snapshots[0], brokerage)
        closed_trade_cursor = 0
        ctx = self._build_context(0, brokerage)
        brokerage.execute_all_deferred(ctx)
        strategy(ctx)
        market_snapshot = self.snapshots[0]
        realized_pnl = brokerage.get_total_realized_pnl(market_snapshot)
        unrealized_pnl = brokerage.get_total_unrealized_pnl(market_snapshot)
        equity_snapshot = self._record_equity_snapshot(
            t=0,
            brokerage=brokerage,
            market_snapshot=market_snapshot,
        )
        closed_trades, closed_trade_cursor = self._collect_new_closed_trades(
            brokerage,
            closed_trade_cursor,
        )
        self.simulation_steps.append(
            SimulationStep(
                t=1,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
                timestamp=equity_snapshot["timestamp"],
                balance=equity_snapshot["balance"],
                cash=equity_snapshot["cash"],
                position_value=equity_snapshot["position_value"],
                equity=equity_snapshot["equity"],
                closed_trades=closed_trades,
            )
        )
        self._build_daily_snapshots()
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
        self.equity_curve = []
        self.closed_trades = []
        self.daily_snapshots = []

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
                collect_equity_curve=self.collect_equity_curve,
                collect_closed_trades=self.collect_closed_trades,
                collect_daily_snapshots=self.collect_daily_snapshots,
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
                    daily_pnl_amount=(
                        session_backtester.daily_snapshots[-1]["daily_pnl_amount"]
                        if session_backtester.daily_snapshots else None
                    ),
                    daily_pnl_pct=(
                        session_backtester.daily_snapshots[-1]["daily_pnl_pct"]
                        if session_backtester.daily_snapshots else None
                    ),
                )
            )
            if self.collect_equity_curve and session_backtester.equity_curve:
                self.equity_curve.extend(session_backtester.equity_curve)
            if self.collect_closed_trades and session_backtester.closed_trades:
                self.closed_trades.extend(session_backtester.closed_trades)
            if self.collect_daily_snapshots and session_backtester.daily_snapshots:
                self.daily_snapshots.extend(session_backtester.daily_snapshots)

        if self.collect_daily_snapshots and self.daily_snapshots:
            self.daily_snapshots = sorted(self.daily_snapshots, key=lambda row: str(row.get("date", "")))

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