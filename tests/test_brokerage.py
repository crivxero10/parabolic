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

    def test_unrealized_pnl_one_asset(self):
        asset_name = "SPY"
        opening_price = 650.00
        operations = [
            Operation(operation_type="BUY", asset_name=asset_name, cost_basis=opening_price) for _ in range(100)
        ]
        b = Brokerage(balance=0.0, positions={"SPY":100}, operations=operations)
        assert b.get_total_unrealized_pnl(market_snapshot={"SPY": 666.66}) == 1666.00

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

    def test_realized_pnl_pct(self):
        operations = [
            Operation(operation_type="BUY", asset_name="MSFT", cost_basis=350.00) for _ in range(100)
        ] + [
            Operation(operation_type="SELL", asset_name="MSFT", cost_basis=395.00) for _ in range(100)
        ]
        b = Brokerage(balance=0.0, positions={"MSFT":100}, operations=operations)

        assert b.get_realized_pnl_pct(market_snapshot={"MSFT": 400.00}) == 0.1429

    def test_unrealized_pnl_pct(self):
        operations = [
            Operation(operation_type="BUY", asset_name="MSFT", cost_basis=350.00) for _ in range(100)
        ]
        b = Brokerage(balance=0.0, positions={"MSFT":100}, operations=operations)
        assert b.get_unrealized_pnl_pct(market_snapshot={"MSFT": 300.00}) == -0.1429

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
        assert b.execute(asset_name="MSFT", units=100, price=420.99)
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
        
    def test_execute_all_deferred_failure(self):
        
        b = Brokerage(balance=10000000.0)
        assert b.execute(asset_name="MSFT", units=100, price=420.99)
        stop_loss_target_price = 400.99

        def stop_loss(ctx: TradingContext):

            current_price = ctx.market[ctx.t]["MSFT"] 
            if current_price <= stop_loss_target_price:
                return True
            return False

        assert b.defer(asset_name="MSFT", units=-100, target_price=stop_loss_target_price, activate=stop_loss)
        assert b.execute(asset_name="MSFT", units=-100, price=stop_loss_target_price)
        assert len(b.execute_all_deferred(
            TradingContext(1, [{"MSFT": 399.00}, {"MSFT": 398.00}, {"MSFT": 399.00}])
            )) == 0
        assert len(b.deferred_instructions) == 0

    def test_execute_all_deferred_pending(self):
        
        b = Brokerage(balance=10000000.0)
        assert b.execute(asset_name="MSFT", units=100, price=420.99)
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