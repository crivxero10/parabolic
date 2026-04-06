import unittest
from typing import Callable
import math
from parabolic.backtest import Backtester, SimulationStep
from parabolic.orchestrator import TradingContext
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
            



    def test_simulation_steps_capture_equity_snapshot_fields(self):
        buying_power = 10000.00
        b = Brokerage(balance=buying_power)
        snapshots = [
            {"SPY": 100.0},
            {"SPY": 100.0},
            {"SPY": 105.0},
        ]

        def buy_once(ctx: TradingContext):
            if ctx.t == 1 and ctx.brokerage.positions.get(ctx.asset_name, 0) == 0:
                assert ctx.brokerage.execute(
                    ctx.asset_name,
                    10,
                    ctx.market[ctx.t][ctx.asset_name],
                    timestamp=f"bar-{ctx.t}",
                )

        backtest = Backtester(snapshots=snapshots, strategy=buy_once, brokerage=b, asset_name="SPY")
        steps = [step for step in backtest]

        assert len(steps) == 3
        assert isinstance(steps[0], SimulationStep)

        # The first yielded step is the historical stale step and should not carry reporting artifacts.
        assert steps[0].timestamp is None
        assert steps[0].balance is None
        assert steps[0].cash is None
        assert steps[0].position_value is None
        assert steps[0].equity is None
        assert steps[0].closed_trades == []

        # After buying 10 units at 100, cash drops but equity remains unchanged.
        assert steps[1].balance == 9000.0
        assert steps[1].cash == 9000.0
        assert steps[1].position_value == 1000.0
        assert steps[1].equity == 10000.0
        assert steps[1].closed_trades == []

        # Mark-to-market on the next bar should only affect position value and equity.
        assert steps[2].balance == 9000.0
        assert steps[2].cash == 9000.0
        assert steps[2].position_value == 1050.0
        assert steps[2].equity == 10050.0
        assert steps[2].closed_trades == []

        # Equity curve should contain one snapshot per simulated bar after the stale step.
        assert len(backtest.equity_curve) == 2
        assert backtest.equity_curve[0]["t"] == 1
        assert backtest.equity_curve[0]["balance"] == 9000.0
        assert backtest.equity_curve[0]["cash"] == 9000.0
        assert backtest.equity_curve[0]["position_value"] == 1000.0
        assert backtest.equity_curve[0]["equity"] == 10000.0
        assert backtest.equity_curve[1]["t"] == 2
        assert backtest.equity_curve[1]["balance"] == 9000.0
        assert backtest.equity_curve[1]["cash"] == 9000.0
        assert backtest.equity_curve[1]["position_value"] == 1050.0
        assert backtest.equity_curve[1]["equity"] == 10050.0

    def test_simulation_collects_closed_trades_once_and_on_exit_step(self):
        buying_power = 10000.00
        b = Brokerage(balance=buying_power)
        snapshots = [
            {"SPY": 100.0},
            {"SPY": 100.0},
            {"SPY": 110.0},
            {"SPY": 110.0},
        ]

        def round_trip(ctx: TradingContext):
            price_now = ctx.market[ctx.t][ctx.asset_name]
            current_units = ctx.brokerage.positions.get(ctx.asset_name, 0)
            if ctx.t == 1 and current_units == 0:
                assert ctx.brokerage.execute(ctx.asset_name, 2, price_now, timestamp="entry-bar")
            elif ctx.t == 2 and current_units > 0:
                assert ctx.brokerage.execute(ctx.asset_name, -2, price_now, timestamp="exit-bar")

        backtest = Backtester(snapshots=snapshots, strategy=round_trip, brokerage=b, asset_name="SPY")
        steps = [step for step in backtest]

        # Two logical one-unit positions should be closed exactly once each.
        assert len(backtest.closed_trades) == 2
        assert steps[1].closed_trades == []
        assert len(steps[2].closed_trades) == 2
        assert steps[3].closed_trades == []
        assert steps[2].closed_trades == backtest.closed_trades

        seen_position_ids = set()
        for trade in backtest.closed_trades:
            assert trade["asset"] == "SPY"
            assert trade["side"] == "long"
            assert trade["entry_timestamp"] == "entry-bar"
            assert trade["exit_timestamp"] == "exit-bar"
            assert trade["entry_price"] == 100.0
            assert trade["exit_price"] == 110.0
            assert trade["quantity"] == 1
            assert trade["pnl_amount"] == 10.0
            assert trade["pnl_pct"] == 0.1
            assert str(trade["position_id"]).startswith("position-")
            assert trade["position_id"] not in seen_position_ids
            seen_position_ids.add(trade["position_id"])

    def test_single_snapshot_session_records_equity_and_closed_trades(self):
        # Start from a consistent state: one existing share bought at 90, with remaining cash 9910.
        b = Brokerage(balance=9910.0, positions={"SPY": 1})
        b.operations = [Operation("BUY", "SPY", 90.0, timestamp="entry-bar", units=1, position_id="position-1")]
        b._rebuild_caches_from_operations()
        snapshots = [{"SPY": 100.0}]

        def sell_all(ctx: TradingContext):
            current_units = ctx.brokerage.positions.get(ctx.asset_name, 0)
            if current_units > 0:
                assert ctx.brokerage.execute(
                    ctx.asset_name,
                    -1 * current_units,
                    ctx.market[ctx.t][ctx.asset_name],
                    timestamp="exit-bar",
                )

        backtest = Backtester(snapshots=snapshots, strategy=sell_all, brokerage=b, asset_name="SPY")
        steps = backtest._simulate_single_snapshot_session(b, sell_all)

        assert len(steps) == 1
        step = steps[0]
        assert step.balance == 10010.0
        assert step.cash == 10010.0
        assert step.position_value == 0.0
        assert step.equity == 10010.0
        assert len(step.closed_trades) == 1
        assert len(backtest.closed_trades) == 1
        assert len(backtest.equity_curve) == 1
        assert backtest.equity_curve[0]["balance"] == 10010.0
        assert backtest.equity_curve[0]["cash"] == 10010.0
        assert backtest.equity_curve[0]["position_value"] == 0.0
        assert backtest.equity_curve[0]["equity"] == 10010.0

        trade = backtest.closed_trades[0]
        assert trade["position_id"] == "position-1"
        assert trade["entry_timestamp"] == "entry-bar"
        assert trade["exit_timestamp"] == "exit-bar"
        assert trade["entry_price"] == 90.0
        assert trade["exit_price"] == 100.0
        assert trade["pnl_amount"] == 10.0
        assert trade["pnl_pct"] == (10.0 / 90.0)

    def test_daily_snapshots_group_by_date_and_compute_daily_pnl(self):
        buying_power = 10000.00
        b = Brokerage(balance=buying_power)
        snapshots = [
            {"SPY": 100.0},
            {"SPY": 101.0},
            {"SPY": 103.0},
        ]

        def buy_once(ctx: TradingContext):
            if ctx.t == 1 and ctx.brokerage.positions.get(ctx.asset_name, 0) == 0:
                assert ctx.brokerage.execute(
                    ctx.asset_name,
                    10,
                    ctx.market[ctx.t][ctx.asset_name],
                    timestamp=f"bar-{ctx.t}",
                )

        backtest = Backtester(snapshots=snapshots, strategy=buy_once, brokerage=b, asset_name="SPY")
        backtest.context_orchestrator.raw_bars = [
            {"t": "2025-01-02T15:59:00Z"},
            {"t": "2025-01-03T15:59:00Z"},
            {"t": "2025-01-03T16:00:00Z"},
        ]
        [step for step in backtest]

        # Only simulated bars contribute snapshots, and both simulated bars fall on 2025-01-03.
        assert len(backtest.daily_snapshots) == 1
        daily_snapshot = backtest.daily_snapshots[0]
        assert daily_snapshot["date"] == "2025-01-03"
        assert daily_snapshot["ending_balance"] == 8990.0
        assert daily_snapshot["ending_cash"] == 8990.0
        assert daily_snapshot["ending_position_value"] == 1030.0
        assert daily_snapshot["ending_equity"] == 10020.0
        # First daily snapshot currently uses starting account balance as its base.
        assert daily_snapshot["daily_pnl_amount"] == 20.0
        assert daily_snapshot["daily_pnl_pct"] == 0.002

    def test_simulate_by_day_single_day_daily_summary_matches_obvious_pnl(self):
        buying_power = 10000.00

        def buy_first_bar(ctx: TradingContext):
            if ctx.brokerage.positions.get(ctx.asset_name, 0) == 0:
                assert ctx.brokerage.execute(
                    ctx.asset_name,
                    10,
                    ctx.market[ctx.t][ctx.asset_name],
                    timestamp=f"bar-{ctx.t}",
                )

        backtest = Backtester(
            snapshots=[{"SPY": 100.0}, {"SPY": 110.0}],
            strategy=buy_first_bar,
            brokerage=Brokerage(balance=buying_power),
            asset_name="SPY",
            timeframe="1Min",
        )
        backtest.context_orchestrator.raw_bars = [
            {"t": "2025-01-02T15:58:00Z"},
            {"t": "2025-01-02T15:59:00Z"},
        ]

        session_results = backtest.simulate_by_day(
            brokerage_factory=lambda: Brokerage(balance=buying_power),
            carry_state=False,
        )

        # Both minute bars belong to the same regular trading session date.
        assert len(session_results) == 1
        session = session_results[0]
        assert session.session_date == "2025-01-02"

        # The strategy buys 10 shares at 100 on the first simulated bar and the day ends at 110.
        # That implies a +100 mark-to-market gain on a 10,000 starting account.
        assert session.daily_pnl_amount == 100.0
        assert session.daily_pnl_pct == 0.01

        # simulate_by_day liquidates remaining positions at session end, so the account finishes fully in cash.
        assert session.end_balance == 10100.0
        assert session.end_available_cash == 10100.0

        # The session is expanded to a full regular trading day, so assert the meaningful invariant:
        # the final step reflects the expected mark-to-market end-of-day equity state before liquidation.
        assert len(session.steps) > 2
        final_step = session.steps[-1]
        assert final_step.equity == 10100.0
        assert final_step.position_value == 1100.0
        assert final_step.cash == 9000.0

    def test_simulate_by_day_splits_sessions_across_dates(self):
        buying_power = 10000.00

        def buy_first_bar(ctx: TradingContext):
            if ctx.brokerage.positions.get(ctx.asset_name, 0) == 0:
                assert ctx.brokerage.execute(
                    ctx.asset_name,
                    10,
                    ctx.market[ctx.t][ctx.asset_name],
                    timestamp=f"bar-{ctx.t}",
                )

        backtest = Backtester(
            snapshots=[{"SPY": 100.0}, {"SPY": 110.0}],
            strategy=buy_first_bar,
            brokerage=Brokerage(balance=buying_power),
            asset_name="SPY",
            timeframe="1Min",
        )
        backtest.context_orchestrator.raw_bars = [
            {"t": "2025-01-02T15:59:00Z"},
            {"t": "2025-01-03T15:59:00Z"},
        ]

        session_results = backtest.simulate_by_day(
            brokerage_factory=lambda: Brokerage(balance=buying_power),
            carry_state=False,
        )

        # One regular-session minute bar on each date should produce two separate daily sessions.
        assert len(session_results) == 2
        assert session_results[0].session_date == "2025-01-02"
        assert session_results[1].session_date == "2025-01-03"

        # Each split session is expanded to a full regular trading day from a single minute seed bar.
        # Since the seeded price is constant within each day, there is no intra-session price change.
        for expected_price, session in zip([100.0, 110.0], session_results):
            assert len(session.steps) > 1
            assert session.daily_pnl_amount == 0.0
            assert session.daily_pnl_pct == 0.0
            # End-of-session liquidation should restore the account to the starting cash when there is no price change.
            assert session.end_balance == 10000.0
            assert session.end_available_cash == 10000.0
            final_step = session.steps[-1]
            assert final_step.cash == 9000.0 if expected_price == 100.0 else 8900.0
            assert final_step.position_value == 10.0 * expected_price
            assert final_step.equity == 10000.0



        
    
            