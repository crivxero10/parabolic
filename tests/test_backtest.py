import unittest
from typing import Callable
import math
from parabolic.backtest import Engine, TradingContext
from parabolic.brokerage import Brokerage, Operation

class TestBacktest(unittest.TestCase):
    
    def test_run_backtest_basic(self):
        
        buying_power = 10000.00
        b = Brokerage(balance=buying_power)
        
        snapshots = [
            {"TLT": price} for price in [101.05, 101.12, 100.09, 100.98, 102.45]
        ]

        e = Engine(snapshots=snapshots)
        # a basic strategy that buys if the price dropped more than a dollar and sells if the price rises more than a dollar
        def simple_active_strategy(ctx: TradingContext):

            price_now = ctx.market[ctx.t]["TLT"]
            price_before = ctx.market[ctx.t-1]["TLT"]

            print(b.positions)

            if price_now - price_before < -1:
                allocation = math.floor(buying_power / price_now)
                b.execute(ctx.asset_name, allocation, price_now)
            elif price_now - price_before > 1 and b.positions[ctx.asset_name] > 0:
                b.execute(ctx.asset_name, -1 * b.positions[ctx.asset_name], price_now)

        assert e.get_rolling_pnl(b, simple_active_strategy) == [0.0, 0.0, 88.11, 233.64] 

    def test_run_backtest_comparison(self):

        buying_power = 10000.00
        bA = Brokerage(balance=buying_power)
        bB = Brokerage(balance=buying_power)
        
        snapshots = [
            {"TLT": price} for price in [101.05, 101.12, 100.09, 100.98, 102.45]
        ]

        e = Engine(snapshots=snapshots)
        
        # a basic strategy that buys if the price dropped more than a dollar and sells if the price rises more than a dollar
        def simple_active_strategy(ctx: TradingContext):

            price_now = ctx.market[ctx.t]["TLT"]
            price_before = ctx.market[ctx.t-1]["TLT"]

            if price_now - price_before < -1:
                allocation = math.floor(buying_power / price_now)
                bA.execute(ctx.asset_name, allocation, price_now)
            elif price_now - price_before > 1 and bA.positions[ctx.asset_name] > 0:
                bA.execute(ctx.asset_name, -1 * bA.positions[ctx.asset_name], price_now)

        # a basic strategy that buys if the price dropped more than a dollar and sells if the price rises more than a dollar
        def simple_passive_strategy(ctx: TradingContext):
            price_now = ctx.market[ctx.t]["TLT"]
            allocation = math.floor(buying_power / price_now)
            if allocation > 0:
                bB.execute(ctx.asset_name, allocation, price_now)

        assert e.get_rolling_pnl(bB, simple_passive_strategy) == [0.0, -100.94, -13.72, 130.34]
        bB = Brokerage(balance=buying_power)
        assert e.get_rolling_pnl_comparison(
            brokerageA=bA, 
            brokerageB=bB, 
            strategyA=simple_active_strategy, 
            strategyB=simple_passive_strategy) == [0, -100.94, -101.83, -103.3]

        