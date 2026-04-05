import unittest
from typing import Callable
import math
from parabolic.backtest import Backtester, TradingContext
from parabolic.brokerage import Brokerage, Operation

class TestBacktest(unittest.TestCase):
    
    def test_run_backtest_basic(self):
        
        buying_power = 10000.00
        b = Brokerage(balance=buying_power)
        
        snapshots = [
            {"TLT": price} for price in [101.05, 101.12, 100.09, 100.98, 102.45]
        ]

        e = Backtester(snapshots=snapshots)
        # a basic strategy that buys if the price dropped more than a dollar and sells if the price rises more than a dollar
        def simple_active_strategy(ctx: TradingContext):
            price_now = ctx.market[ctx.t]["TLT"]
            price_before = ctx.market[ctx.t-1]["TLT"]
            if price_now - price_before < -1:
                allocation = math.floor(buying_power / price_now)
                b.execute(ctx.asset_name, allocation, price_now)
            elif price_now - price_before > 1 and b.positions[ctx.asset_name] > 0:
                b.execute(ctx.asset_name, -1 * b.positions[ctx.asset_name], price_now)

        assert e.get_rolling_pnl(b, simple_active_strategy) == [0.0, 0.0, 0.0, 88.11, 233.64] 

    def test_run_backtest_comparison(self):

        buying_power = 10000.00
        bA = Brokerage(balance=buying_power)
        bB = Brokerage(balance=buying_power)
        
        snapshots = [
            {"TLT": price} for price in [101.05, 101.12, 100.09, 100.98, 102.45]
        ]

        e = Backtester(snapshots=snapshots)
        
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

        assert e.get_rolling_pnl(bB, simple_passive_strategy) == [0.0, 0.0, -100.94, -13.72, 130.34]
        bB = Brokerage(balance=buying_power)
        assert e.get_rolling_pnl_comparison(
            brokerageA=bA, 
            brokerageB=bB, 
            strategyA=simple_active_strategy, 
            strategyB=simple_passive_strategy) == [0, 0, -100.94, -101.83, -103.3]
        
    def test_run_backtest_with_stop_loss(self):

        buying_power = 10000.00
        b = Brokerage(balance=buying_power)
        
        snapshots = [
            {"TLT": price} for price in [101.05, 101.12, 100.09, 95.99, 106.99]
        ]

        e = Backtester(snapshots=snapshots)
        # a basic strategy that buys if the price dropped more than a dollar and sells if the price rises more than a dollar
        def simple_active_strategy_with_stop_loss(ctx: TradingContext):
            price_now = ctx.market[ctx.t]["TLT"]
            price_before = ctx.market[ctx.t-1]["TLT"]
            if price_now - price_before < -1:
                # -2% stop loss
                stop_price = price_now * 0.98
                def stop_loss(at_stop_ctx: TradingContext):
                    current_price = at_stop_ctx.market[at_stop_ctx.t]["TLT"] 
                    if current_price <= stop_price:
                        return True
                    return False
                bp = b.available_cash
                allocation = math.floor(bp / price_now)
                b.execute(ctx.asset_name, allocation, price_now)
                b.defer(ctx.asset_name, -allocation, stop_price, stop_loss)
            elif price_now - price_before > 1 and b.positions[ctx.asset_name] > 0:
                b.execute(ctx.asset_name, -1 * b.positions[ctx.asset_name], price_now)

        assert e.get_rolling_pnl(b, simple_active_strategy_with_stop_loss) == [0.0, 0.0, 0.0, -405.90, 683.10]

    def test_run_backtest_with_stop_loss_settled_cash_only(self):

        buying_power = 10000.00
        b = Brokerage(balance=buying_power, settled_cash_only=True)
        
        snapshots = [
            {"TLT": price} for price in [101.05, 101.12, 100.09, 95.99, 106.99]
        ]

        e = Backtester(snapshots=snapshots)
        # a basic strategy that buys if the price dropped more than a dollar and sells if the price rises more than a dollar
        def simple_active_strategy_with_stop_loss(ctx: TradingContext):
            price_now = ctx.market[ctx.t]["TLT"]
            price_before = ctx.market[ctx.t-1]["TLT"]
            if price_now - price_before < -1:
                # -2% stop loss
                stop_price = price_now * 0.98
                def stop_loss(at_stop_ctx: TradingContext):
                    current_price = at_stop_ctx.market[at_stop_ctx.t]["TLT"] 
                    if current_price <= stop_price:
                        return True
                    return False
                bp = b.available_cash
                allocation = math.floor(bp / price_now)
                b.execute(ctx.asset_name, allocation, price_now)
                b.defer(ctx.asset_name, -allocation, stop_price, stop_loss)
            elif price_now - price_before > 1 and b.positions[ctx.asset_name] > 0:
                b.execute(ctx.asset_name, -1 * b.positions[ctx.asset_name], price_now)

        assert e.get_rolling_pnl(b, simple_active_strategy_with_stop_loss) == [0.0, 0.0, 0.0, -405.90, -405.90]  

    def test_stepper_backtest(self):
        
        buying_power = 10000.00
        b = Brokerage(balance=buying_power, settled_cash_only=True)

        def simple_dca(ctx: TradingContext):
            price_now = ctx.market[ctx.t][ctx.asset_name]
            current_dca = b.get_avg_cost_basis(ctx.asset_name)
            if price_now < current_dca or current_dca == 0:
                bp = b.available_cash
                allocation = math.floor(bp * 0.10 / price_now)
                if allocation > 0:
                    b.execute(ctx.asset_name, allocation, price_now)

        # 4 months of fictional SPY
        SPY = [
            461.80, 400, 390, 380
        ]
        snapshots = [
            {"SPY": price} for price in SPY
        ]
        spy_backtest = Backtester(snapshots=snapshots, strategy=simple_dca, brokerage=b, asset_name="SPY")
        steps = [step for step in spy_backtest]

        assert len(steps) == 4
        assert steps[0].unrealized_pnl == 0
        assert steps[1].unrealized_pnl == 0
        assert steps[2].unrealized_pnl == -20.0
        assert steps[3].unrealized_pnl == -60.0
            



        
    
            