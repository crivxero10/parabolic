import unittest
from typing import Callable
import math
from parabolic.brokerage import Brokerage, Operation

class TradingContext:
    def __init__(self, t:int, snapshot: list[dict[str, float]], asset_name: str = "TLT"):
        self.market = snapshot
        self.asset_name = asset_name
        self.t = t  # current index
    
class Engine:
    def __init__(self, snapshots: list[dict[str, float]], asset_name: str = "TLT"):
        self.snapshots = snapshots
        self.asset_name = asset_name

    def get_rolling_pnl(
    self,
    brokerage: Brokerage,
    strategy: Callable[[TradingContext], None],
    ) -> list[float]:
        
        pnls: list[float] = []

        for t in range(1, len(self.snapshots)):

            ctx = TradingContext(t=t, snapshot=self.snapshots[:t+1], asset_name=self.asset_name)
            strategy(ctx)

            # compute pnl using full snapshot
            unrealized_pnl = brokerage.get_total_unrealized_pnl(self.snapshots[t])
            realized_pnl = brokerage.get_total_realized_pnl(self.snapshots[t])

            total_pnl = unrealized_pnl + realized_pnl
            print(brokerage.positions)
            print(unrealized_pnl, realized_pnl, total_pnl)

            pnls.append(total_pnl)
        return pnls
    
    def get_rolling_pnl_comparison(
    self,
    brokerageA: Brokerage,
    brokerageB: Brokerage,
    strategyA: Callable[[TradingContext], None],
    strategyB: Callable[[TradingContext], None],
    ) -> list[float]:
        'B-A'
        pnlA = self.get_rolling_pnl(brokerageA, strategyA)
        pnlB = self.get_rolling_pnl(brokerageB, strategyB)

        print(pnlA, pnlB)

        return [round(b - a, 2) for a, b in zip(pnlA, pnlB)]
        