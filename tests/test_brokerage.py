import unittest
from parabolic.brokerage import Brokerage, Operation
from parabolic.backtest import TradingContext

class TestBrokerage(unittest.TestCase):
    
    def test_open_trade_success(self):
        initial_balance = 99999.99
        b = Brokerage(balance=initial_balance)
        asset_name = "SPY"
        units = 100
        price = 666.66
        assert b.execute(asset_name=asset_name, units=units, price=price) == True
        assert b.positions[asset_name] == units
        assert len(b.operations) == units
        assert b.balance == initial_balance - units * price
        for _, o in enumerate(b.operations):
            assert o.operation_type == "BUY"
            assert o.asset_name == asset_name
            assert o.cost_basis == price

    def test_open_trade_failure(self):
        b = Brokerage(balance=99999.99)
        asset_name = "SPY"
        units = 200
        price = 666.66
        assert b.execute(asset_name=asset_name, units=units, price=price) == False

    def test_close_trade_failure(self):
        b = Brokerage(balance=0.0, positions={"SPY":99})
        asset_name = "SPY"
        # tries to sell 100
        units = -100
        price = 699.99
        assert b.execute(asset_name=asset_name, units=units, price=price) == False
        
    def test_close_trade_success(self):
        b = Brokerage(balance=0.0, positions={"SPY":100})
        asset_name = "SPY"
        # tries to sell 100
        units = -100
        price = 699.99
        assert b.execute(asset_name=asset_name, units=units, price=price) == True
        assert b.balance == -1 * units * price

    def test_closed_trades_single_round_trip(self):
        b = Brokerage(balance=100000.0)
        assert b.execute(asset_name="SPY", units=2, price=100.0, timestamp="2025-01-02T10:00:00Z")
        assert b.execute(asset_name="SPY", units=-2, price=110.0, timestamp="2025-01-03T11:00:00Z")

        closed_trades = b.get_closed_trades()
        assert len(closed_trades) == 2
        for trade in closed_trades:
            assert trade["asset"] == "SPY"
            assert trade["side"] == "long"
            assert trade["entry_timestamp"] == "2025-01-02T10:00:00Z"
            assert trade["exit_timestamp"] == "2025-01-03T11:00:00Z"
            assert trade["entry_price"] == 100.0
            assert trade["exit_price"] == 110.0
            assert trade["quantity"] == 1
            assert trade["pnl_amount"] == 10.0
            assert trade["pnl_pct"] == 0.1
            assert str(trade["position_id"]).startswith("position-")

    def test_execution_log_contains_buy_and_sell_records(self):
        b = Brokerage(balance=100000.0)
        assert b.execute(asset_name="MSFT", units=1, price=200.0, timestamp="2025-01-02T10:00:00Z")
        assert b.execute(asset_name="MSFT", units=-1, price=210.0, timestamp="2025-01-02T15:30:00Z")

        execution_log = b.get_execution_log()
        assert len(execution_log) == 2

        buy_log = execution_log[0]
        assert buy_log["operation_type"] == "BUY"
        assert buy_log["asset"] == "MSFT"
        assert buy_log["price"] == 200.0
        assert buy_log["units"] == 1
        assert buy_log["timestamp"] == "2025-01-02T10:00:00Z"
        assert buy_log["pnl_amount"] == 0.0
        assert buy_log["pnl_pct"] == 0.0
        assert str(buy_log["position_id"]).startswith("position-")

        sell_log = execution_log[1]
        assert sell_log["operation_type"] == "SELL"
        assert sell_log["asset"] == "MSFT"
        assert sell_log["price"] == 210.0
        assert sell_log["units"] == 1
        assert sell_log["timestamp"] == "2025-01-02T15:30:00Z"
        assert sell_log["pnl_amount"] == 10.0
        assert sell_log["pnl_pct"] == 0.05
        assert sell_log["position_id"] == buy_log["position_id"]

    def test_order_log_aliases_execution_log(self):
        b = Brokerage(balance=100000.0)
        assert b.execute(asset_name="QQQ", units=1, price=300.0, timestamp="2025-01-02T10:00:00Z")

        assert b.get_order_log() == b.get_execution_log()

    def test_operations_capture_timestamp_units_and_position_id(self):
        b = Brokerage(balance=100000.0)
        assert b.execute(asset_name="NVDA", units=2, price=150.0, timestamp="2025-01-02T10:00:00Z")

        assert len(b.operations) == 2
        for operation in b.operations:
            assert operation.operation_type == "BUY"
            assert operation.asset_name == "NVDA"
            assert operation.cost_basis == 150.0
            assert operation.timestamp == "2025-01-02T10:00:00Z"
            assert operation.units == 1
            assert str(operation.position_id).startswith("position-")
            operation_dict = operation.to_dict()
            assert operation_dict["timestamp"] == "2025-01-02T10:00:00Z"
            assert operation_dict["units"] == 1
            assert str(operation_dict["position_id"]).startswith("position-")

    def test_rebuild_caches_restores_closed_trades_and_execution_log(self):
        operations = [
            Operation(
                operation_type="BUY",
                asset_name="SPY",
                cost_basis=100.0,
                timestamp="2025-01-02T10:00:00Z",
                units=1,
                position_id="position-1",
            ),
            Operation(
                operation_type="SELL",
                asset_name="SPY",
                cost_basis=110.0,
                timestamp="2025-01-03T11:00:00Z",
                units=1,
                position_id="position-1",
            ),
        ]
        b = Brokerage(balance=10.0, positions={"SPY": 0}, operations=operations)

        closed_trades = b.get_closed_trades()
        assert len(closed_trades) == 1
        trade = closed_trades[0]
        assert trade["position_id"] == "position-1"
        assert trade["asset"] == "SPY"
        assert trade["entry_timestamp"] == "2025-01-02T10:00:00Z"
        assert trade["exit_timestamp"] == "2025-01-03T11:00:00Z"
        assert trade["pnl_amount"] == 10.0
        assert trade["pnl_pct"] == 0.1

        execution_log = b.get_execution_log()
        assert len(execution_log) == 2
        assert execution_log[0]["operation_type"] == "BUY"
        assert execution_log[1]["operation_type"] == "SELL"
        assert execution_log[1]["pnl_amount"] == 10.0
        assert execution_log[1]["pnl_pct"] == 0.1

    def test_unrealized_pnl_one_asset(self):
        asset_name = "SPY"
        opening_price = 650.00
        operations = [
            Operation(operation_type="BUY", asset_name=asset_name, cost_basis=opening_price) for _ in range(100)
        ]
        b = Brokerage(balance=0.0, positions={"SPY":100}, operations=operations)
        assert b.get_total_unrealized_pnl(market_snapshot={"SPY": 666.66}) == 1666.00

    def test_unrealized_pnl_multiple_operations(self):
        asset_name = "SPY"
        opening_price = 650.00
        closing_price = 600.00
        operations = [
            Operation(operation_type="BUY", asset_name=asset_name, cost_basis=opening_price) for _ in range(100)
        ] + [
            Operation(operation_type="SELL", asset_name=asset_name, cost_basis=closing_price) for _ in range(100)
        ] + [
            Operation(operation_type="BUY", asset_name=asset_name, cost_basis=closing_price) for _ in range(100)
        ]
        b = Brokerage(balance=0.0, positions={"SPY":100}, operations=operations)
        assert b.get_total_unrealized_pnl(market_snapshot={"SPY": 600.00}) == 0

    def test_unrealized_pnl_dca(self):
        asset_name = "SPY"
        first_opening_price = 650.00
        second_opening_price = 640.00
        third_opening_price = 660.00
        operations = [
            Operation(operation_type="BUY", asset_name=asset_name, cost_basis=first_opening_price) for _ in range(100)
        ] + [
            Operation(operation_type="BUY", asset_name=asset_name, cost_basis=second_opening_price) for _ in range(100)
        ] + [
            Operation(operation_type="BUY", asset_name=asset_name, cost_basis=third_opening_price) for _ in range(100)
        ] 
        b = Brokerage(balance=0.0, positions={"SPY":100}, operations=operations)
        assert b.get_total_unrealized_pnl(market_snapshot={"SPY": 650.00}) == 0
        assert b.get_total_unrealized_pnl(market_snapshot={"SPY": 660.00}) == 1000
        assert b.get_total_unrealized_pnl(market_snapshot={"SPY": 640.00}) == -1000

    def test_unrealized_pnl_multi_asset(self):
        operations = [
            Operation(operation_type="BUY", asset_name="SPY", cost_basis=610.00) for _ in range(100)
        ] + [
            Operation(operation_type="BUY", asset_name="QQQ", cost_basis=600.00) for _ in range(400)
        ]
        b = Brokerage(balance=0.0, positions={"SPY":100, "QQQ": 400}, operations=operations)
        assert b.get_total_unrealized_pnl(market_snapshot={"SPY": 666.66, "QQQ": 512.00}) == -29534

    def test_realized_pnl_one_asset(self):
        operations = [
            Operation(operation_type="BUY", asset_name="TLT", cost_basis=150.00) for _ in range(400)
        ] + [
            Operation(operation_type="SELL", asset_name="TLT", cost_basis=80.00) for _ in range(200)
        ]
        b = Brokerage(balance=0.0, positions={"TLT":200}, operations=operations)
        assert b.get_total_realized_pnl(market_snapshot={"TLT": 95.00}) == -14000

    def test_realized_pnl_multi_asset(self):
        operations = [
            Operation(operation_type="BUY", asset_name="TLT", cost_basis=150.00) for _ in range(400)
        ] + [
            Operation(operation_type="SELL", asset_name="TLT", cost_basis=80.00) for _ in range(200)
        ] + [
            Operation(operation_type="BUY", asset_name="BIL", cost_basis=91.00) for _ in range(50)
        ] + [
            Operation(operation_type="SELL", asset_name="BIL", cost_basis=89.00) for _ in range(20)
        ]
        b = Brokerage(balance=0.0, positions={"TLT":200, "BIL":30}, operations=operations)
        assert b.get_total_realized_pnl(market_snapshot={"TLT": 95.00, "BIL": 91.00}) == -14040

    def test_get_total_pnl(self):
        operations = [
            Operation(operation_type="BUY", asset_name="TLT", cost_basis=150.00) for _ in range(400)
        ] + [
            Operation(operation_type="SELL", asset_name="TLT", cost_basis=80.00) for _ in range(200)
        ] + [
            Operation(operation_type="BUY", asset_name="BIL", cost_basis=91.00) for _ in range(50)
        ] + [
            Operation(operation_type="SELL", asset_name="BIL", cost_basis=89.00) for _ in range(20)
        ]
        b = Brokerage(balance=0.0, positions={"TLT":200, "BIL":30}, operations=operations)

        realized = b.get_total_realized_pnl(market_snapshot={"TLT": 95.00}) 
        unrealized = b.get_total_unrealized_pnl(market_snapshot={"TLT": 95.00, "BIL": 91.00}) 
        assert realized + unrealized == -25040

    def test_realized_pnl_pct_1(self):
        operations = [
            Operation(operation_type="BUY", asset_name="MSFT", cost_basis=350.00) for _ in range(100)
        ] + [
            Operation(operation_type="SELL", asset_name="MSFT", cost_basis=395.00) for _ in range(100)
        ]
        b = Brokerage(balance=0.0, positions={"MSFT":100}, operations=operations)

        assert b.get_realized_pnl_pct(market_snapshot={"MSFT": 400.00}) == 0.1286

    def test_realized_pnl_pct_2(self):
        operations = [
            Operation(operation_type="BUY", asset_name="MSFT", cost_basis=350.00) for _ in range(100)
        ]
        b = Brokerage(balance=0.0, positions={"MSFT":100}, operations=operations)

        assert b.get_realized_pnl_pct(market_snapshot={"MSFT": 400.00}) == 0

    def test_unrealized_pnl_pct_1(self):
        operations = [
            Operation(operation_type="BUY", asset_name="MSFT", cost_basis=350.00) for _ in range(100)
        ]
        b = Brokerage(balance=0.0, positions={"MSFT":100}, operations=operations)
        assert b.get_unrealized_pnl_pct(market_snapshot={"MSFT": 300.00}) == -0.1429

    def test_unrealized_pnl_pct_2(self):
        operations = [
            Operation(operation_type="BUY", asset_name="MSFT", cost_basis=350.00) for _ in range(100)
        ] + [
            Operation(operation_type="SELL", asset_name="MSFT", cost_basis=395.00) for _ in range(100)
        ]
        b = Brokerage(balance=0.0, positions={"MSFT":100}, operations=operations)

        assert b.get_unrealized_pnl_pct(market_snapshot={"MSFT": 400.00}) == 0.0

    def test_can_defer_order(self):
        
        b = Brokerage(balance=10000000.0)
        assert b.execute(asset_name="MSFT", units=100, price=420.99)
        stop_loss_target_price = 400.99

        def stop_loss(ctx: TradingContext):

            current_price = ctx.market[ctx.t]["MSFT"] 
            if current_price <= stop_loss_target_price:
                return True
            return False

        assert b.defer(asset_name="MSFT", units=-100, target_price=stop_loss_target_price, activate=stop_loss)
        assert b.deferred_instructions[0].activate(
            TradingContext(1, [{"MSFT": 399.00}, {"MSFT": 398.00}, {"MSFT": 399.00}])
        ) == True

    def test_cannot_defer_order(self):
        b = Brokerage(balance=10000000.0)
        stop_loss_target_price = 400.99

        def stop_loss(ctx: TradingContext):

            current_price = ctx.market[ctx.t]["MSFT"] 
            if current_price <= stop_loss_target_price:
                return True
            return False

        assert not b.defer(asset_name="MSFT", units=-100, target_price=stop_loss_target_price, activate=stop_loss)

    def test_execute_all_deferred_success(self):
        
        b = Brokerage(balance=10000000.0)
        assert b.execute(asset_name="MSFT", units=100, price=420.99, timestamp="2025-01-02T09:30:00Z")
        stop_loss_target_price = 400.99

        def stop_loss(ctx: TradingContext):

            current_price = ctx.market[ctx.t]["MSFT"] 
            if current_price <= stop_loss_target_price:
                return True
            return False

        assert b.defer(asset_name="MSFT", units=-100, target_price=stop_loss_target_price, activate=stop_loss)
        assert len(b.execute_all_deferred(
            TradingContext(1, [{"MSFT": 399.00}, {"MSFT": 398.00}, {"MSFT": 399.00}])
            )) == 1
        
        closed_trades = b.get_closed_trades()
        assert len(closed_trades) == 100
        for trade in closed_trades:
            assert trade["entry_timestamp"] == "2025-01-02T09:30:00Z"
            assert trade["exit_timestamp"] == "1"
            assert trade["asset"] == "MSFT"
            assert trade["exit_price"] == 398.00

    def test_execute_all_deferred_failure(self):
        
        b = Brokerage(balance=10000000.0)
        assert b.execute(asset_name="MSFT", units=100, price=420.99, timestamp="2025-01-02T09:30:00Z")
        stop_loss_target_price = 400.99

        def stop_loss(ctx: TradingContext):

            current_price = ctx.market[ctx.t]["MSFT"] 
            if current_price <= stop_loss_target_price:
                return True
            return False

        assert b.defer(asset_name="MSFT", units=-100, target_price=stop_loss_target_price, activate=stop_loss)
        assert b.execute(asset_name="MSFT", units=-100, price=stop_loss_target_price, timestamp="2025-01-02T10:00:00Z")
        assert len(b.execute_all_deferred(
            TradingContext(1, [{"MSFT": 399.00}, {"MSFT": 398.00}, {"MSFT": 399.00}])
            )) == 0
        assert len(b.deferred_instructions) == 0

    def test_execute_all_deferred_pending(self):
        
        b = Brokerage(balance=10000000.0)
        assert b.execute(asset_name="MSFT", units=100, price=420.99, timestamp="2025-01-02T09:30:00Z")
        stop_loss_target_price = 400.99

        def stop_loss(ctx: TradingContext):

            current_price = ctx.market[ctx.t]["MSFT"] 
            if current_price <= stop_loss_target_price:
                return True
            return False

        assert b.defer(asset_name="MSFT", units=-100, target_price=stop_loss_target_price, activate=stop_loss)
        assert len(b.execute_all_deferred(
            TradingContext(1, [{"MSFT": 430.00}, {"MSFT": 440.00}, {"MSFT": 445.00}])
            )) == 0
        assert len(b.deferred_instructions) == 1

    def test_settled_cash_only(self):
        b = Brokerage(balance=4200, settled_cash_only=True)
        assert b.execute(asset_name="MSFT", units=10, price=420)
        assert b.execute(asset_name="MSFT", units=-10, price=440)
        assert not b.execute(asset_name="MSFT", units=10, price=420)

    def test_cost_average_open_positions(self):
        b = Brokerage(balance=999999, settled_cash_only=True)
        assert b.execute(asset_name="MSFT", units=10, price=420)
        assert b.execute(asset_name="MSFT", units=10, price=440)
        assert b.execute(asset_name="MSFT", units=30, price=500)
        assert b.get_avg_cost_basis(asset_name="MSFT") == 472.00

    def test_cost_average_no_positions(self):
        b = Brokerage(balance=999999, settled_cash_only=True)
        assert b.execute(asset_name="MSFT", units=10, price=420)
        assert b.execute(asset_name="MSFT", units=10, price=440)
        assert b.execute(asset_name="MSFT", units=30, price=500)
        assert b.get_avg_cost_basis(asset_name="NVDA") == 0





