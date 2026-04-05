import unittest
from typing import Callable
import math
from parabolic.brokerage import Brokerage, Operation

class TradingContext:

    def __init__(self, t:int, snapshot: list[dict[str, float]], asset_name: str = "TLT"):
        self.market = snapshot
        self.asset_name = asset_name
        self.t = t  # current index

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

class Backtester:
    def __init__(
            self,
            snapshots: list[dict[str, float]],
            strategy: Callable[[TradingContext], None] | None = None,
            brokerage: Brokerage | None = None,
            asset_name: str = "TLT"):
        
        self.snapshots = snapshots
        self.strategy = strategy
        self.brokerage = brokerage
        self.asset_name = asset_name
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
            raise ValueError("Backtester iterator is not configured. Pass a brokerage and strategy at construction time, call simulate(...)/iter_simulation(...), or set iteration inputs before iterating.")
        return self.iter_simulation(brokerage, strategy)

    def _resolve_strategy(self, strategy: Callable[[TradingContext], None] | None) -> Callable[[TradingContext], None]:
        resolved_strategy = strategy or self.strategy
        if resolved_strategy is None:
            raise ValueError("No strategy configured. Pass a strategy at construction time or to the method call.")
        return resolved_strategy

    def iter_simulation(
        self,
        brokerage: Brokerage,
        strategy: Callable[[TradingContext], None] | None = None,
    ):
        strategy = self._resolve_strategy(strategy)
        self._iter_brokerage = brokerage
        self._iter_strategy = strategy
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

    def _build_context(self, t: int) -> TradingContext:
        return TradingContext(t=t, snapshot=self.snapshots[:t + 1], asset_name=self.asset_name)

    def _simulate_step(
        self,
        t: int,
        brokerage: Brokerage,
        strategy: Callable[[TradingContext], None],
    ) -> SimulationStep:
        ctx = self._build_context(t)
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

    def simulate(
        self,
        brokerage: Brokerage,
        strategy: Callable[[TradingContext], None] | None = None,
    ) -> list[SimulationStep]:
        for _ in self.iter_simulation(brokerage, strategy):
            pass
        return self.simulation_steps

    def get_rolling_pnl(
        self,
        brokerage: Brokerage,
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
        
        'B-A'
        pnlA = self.get_rolling_pnl(brokerageA, strategyA)
        pnlB = self.get_rolling_pnl(brokerageB, strategyB)
        return [round(b - a, 2) for a, b in zip(pnlA, pnlB)]