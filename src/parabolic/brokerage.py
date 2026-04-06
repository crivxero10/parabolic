from collections import deque
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from parabolic.backtest import TradingContext

class Operation:

    def __init__(self, operation_type: str, asset_name: str, cost_basis: float):
        self.operation_type = operation_type
        self.asset_name = asset_name
        self.cost_basis = cost_basis

    def __str__(self):
        return f"{self.operation_type} {self.asset_name} @ {self.cost_basis}"
    
    def __repr__(self):
        return f"{self.operation_type} {self.asset_name} @ {self.cost_basis}"   
    
class Instruction:

    def __init__(self, asset_name: str, units: int, target_price: float, activate: Callable[..., bool]):
        self.asset_name = asset_name
        self.units = units
        self.target_price = target_price
        self.activate = activate

    def __str__(self):
        side = "BUY" if self.units > 0 else "SELL" if self.units < 0 else "HOLD"
        return f"{side} {self.asset_name} {abs(self.units)} @ {self.target_price}"
    
    def __repr__(self):
        return self.__str__()


class Brokerage:

    def __init__(
            self, 
            balance: float,
            available_cash: float | None = None,
            positions: dict[str, int] | None = None, 
            operations: list[Operation] | None = None,
            deferred_instructions: list[Instruction] | None = None,
            settled_cash_only: bool | None = None):
        
        self.balance = balance
        self.available_cash = balance if available_cash is None else available_cash
        self.positions = positions or {}
        self.operations = operations or []
        self.deferred_instructions = deferred_instructions or []
        self.settled_cash_only = settled_cash_only or False
        self.deposit_history = [balance]
        self._inventory_lots: dict[str, deque[list[float | int]]] = {}
        self._realized_pnl_total = 0.0
        self._realized_cost_total = 0.0
        self._rebuild_caches_from_operations()

    def _rebuild_caches_from_operations(self) -> None:
        self._inventory_lots = {}
        self._realized_pnl_total = 0.0
        self._realized_cost_total = 0.0
        for op in self.operations:
            if op.operation_type == "BUY":
                self._record_buy(op.asset_name, op.cost_basis, 1)
            elif op.operation_type == "SELL":
                self._record_sell(op.asset_name, op.cost_basis, 1)

    def _record_buy(self, asset_name: str, price: float, units: int) -> None:
        if units <= 0:
            return
        lots = self._inventory_lots.setdefault(asset_name, deque())
        if lots and lots[-1][0] == price:
            lots[-1][1] += units
        else:
            lots.append([price, units])

    def _record_sell(self, asset_name: str, price: float, units: int) -> None:
        if units <= 0:
            return
        lots = self._inventory_lots.setdefault(asset_name, deque())
        remaining_units = units
        while remaining_units > 0 and lots:
            buy_price, buy_units = lots[0]
            matched_units = min(remaining_units, int(buy_units))
            self._realized_pnl_total += matched_units * (price - float(buy_price))
            self._realized_cost_total += matched_units * float(buy_price)
            buy_units = int(buy_units) - matched_units
            remaining_units -= matched_units
            if buy_units == 0:
                lots.popleft()
            else:
                lots[0][1] = buy_units

    def _append_operations(self, operation_type: str, asset_name: str, price: float, units: int) -> None:
        if units <= 0:
            return
        self.operations.extend(
            Operation(operation_type, asset_name, price)
            for _ in range(units)
        )

    def _get_open_lots_for_position(self, asset_name: str, units: int) -> list[tuple[float, int]]:
        if units <= 0:
            return []
        lots = self._inventory_lots.get(asset_name, deque())
        remaining_units = units
        limited_lots: list[tuple[float, int]] = []
        for buy_price, buy_units in lots:
            if remaining_units <= 0:
                break
            matched_units = min(remaining_units, int(buy_units))
            if matched_units > 0:
                limited_lots.append((float(buy_price), matched_units))
                remaining_units -= matched_units
        return limited_lots

    def execute(self, asset_name: str, units: int, price: float) -> bool:
        # BUY
        if units > 0:
            total_cost = units * price
            if total_cost > self.available_cash:
                return False
            self.balance -= total_cost
            self.available_cash -= total_cost
            if asset_name not in self.positions:
                self.positions[asset_name] = 0
            self.positions[asset_name] += units
            self._record_buy(asset_name, price, units)
            self._append_operations("BUY", asset_name, price, units)
            return True
        # SELL
        if units < 0:
            sell_units = abs(units)
            sale_proceeds = sell_units * price
            # not enough position
            if asset_name not in self.positions or self.positions[asset_name] < sell_units:
                return False
            self.positions[asset_name] -= sell_units
            self.balance += sale_proceeds
            if not self.settled_cash_only:
                self.available_cash += sale_proceeds
            self._record_sell(asset_name, price, sell_units)
            self._append_operations("SELL", asset_name, price, sell_units)
            return True
        return False
    
    def defer(self, asset_name: str, units: int, target_price: float, activate: Callable[..., bool]) -> bool:
        reserved_balance = 0.0
        reserved_positions: dict[str, int] = {}
        for instruction in self.deferred_instructions:
            reserved_balance = self._reserve_instruction(
                instruction,
                reserved_balance,
                reserved_positions,
            )
        if not self._is_instruction_compatible(
            asset_name,
            units,
            target_price,
            reserved_balance,
            reserved_positions,
        ):
            return False
        self.deferred_instructions.append(
            Instruction(
                asset_name=asset_name,
                units=units,
                target_price=target_price,
                activate=activate,
            )
        )
        return True
    
    def execute_all_deferred(self, ctx: TradingContext) -> list[Instruction]:
        executed_instructions: list[Instruction] = []
        remaining_instructions: list[Instruction] = []
        current_market = ctx.market[ctx.t]
        reserved_balance = 0.0
        reserved_positions: dict[str, int] = {}
        for instruction in self.deferred_instructions:
            if not self._is_instruction_compatible(
                instruction.asset_name,
                instruction.units,
                instruction.target_price,
                reserved_balance,
                reserved_positions,
            ):
                continue
            should_activate = instruction.activate(ctx)
            if should_activate:
                if instruction.asset_name in current_market and self.execute(
                    asset_name=instruction.asset_name,
                    units=instruction.units,
                    price=current_market[instruction.asset_name],
                ):
                    executed_instructions.append(instruction)
            else:
                remaining_instructions.append(instruction)
                reserved_balance = self._reserve_instruction(
                    instruction,
                    reserved_balance,
                    reserved_positions,
                )
        self.deferred_instructions = remaining_instructions
        return executed_instructions
    
    def _is_instruction_compatible(
        self,
        asset_name: str,
        units: int,
        target_price: float,
        reserved_balance: float,
        reserved_positions: dict[str, int],
    ) -> bool:
        if units == 0:
            return False
        if units > 0:
            required_balance = units * target_price
            available_balance = self.available_cash - reserved_balance
            return required_balance <= available_balance
        required_units = abs(units)
        available_units = self.positions.get(asset_name, 0) - reserved_positions.get(asset_name, 0)
        return required_units <= available_units

    def _reserve_instruction(
        self,
        instruction: Instruction,
        reserved_balance: float,
        reserved_positions: dict[str, int],
    ) -> float:
        if instruction.units > 0:
            return reserved_balance + (instruction.units * instruction.target_price)
        if instruction.units < 0:
            reserved_positions[instruction.asset_name] = (
                reserved_positions.get(instruction.asset_name, 0) + abs(instruction.units)
            )
        return reserved_balance

    def _get_open_inventory(self) -> dict[str, list[float]]:
        inventory: dict[str, list[float]] = {}
        for asset_name, lots in self._inventory_lots.items():
            inventory[asset_name] = []
            for buy_price, buy_units in lots:
                inventory[asset_name].extend([float(buy_price)] * int(buy_units))
        return inventory

    def get_avg_cost_basis(self, asset_name: str) -> float:
        units = self.positions.get(asset_name, 0)
        if units <= 0:
            return 0.0

        open_lots = self._get_open_lots_for_position(asset_name, units)
        if not open_lots:
            return 0.0

        total_units = sum(lot_units for _, lot_units in open_lots)
        if total_units == 0:
            return 0.0

        total_cost = sum(price * lot_units for price, lot_units in open_lots)
        return total_cost / total_units

    def _get_realized_matches(self) -> list[tuple[float, float]]:
        matches: list[tuple[float, float]] = []
        inventory: dict[str, deque[list[float | int]]] = {}
        for op in self.operations:
            asset = op.asset_name
            inventory.setdefault(asset, deque())
            if op.operation_type == "BUY":
                lots = inventory[asset]
                if lots and lots[-1][0] == op.cost_basis:
                    lots[-1][1] += 1
                else:
                    lots.append([op.cost_basis, 1])
            elif op.operation_type == "SELL":
                lots = inventory[asset]
                remaining_units = 1
                while remaining_units > 0 and lots:
                    buy_price, buy_units = lots[0]
                    matched_units = min(remaining_units, int(buy_units))
                    matches.extend([(float(buy_price), op.cost_basis)] * matched_units)
                    buy_units = int(buy_units) - matched_units
                    remaining_units -= matched_units
                    if buy_units == 0:
                        lots.popleft()
                    else:
                        lots[0][1] = buy_units
        return matches

    def get_total_unrealized_pnl(self, market_snapshot: dict[str, float]) -> float:
        pnl = 0.0
        for asset_name, units in self.positions.items():
            if units <= 0 or asset_name not in market_snapshot:
                continue
            for buy_price, lot_units in self._get_open_lots_for_position(asset_name, units):
                pnl += lot_units * (market_snapshot[asset_name] - buy_price)
        return round(pnl, 2)
    
    def get_total_realized_pnl(self, market_snapshot: dict[str, float]) -> float:
        return round(self._realized_pnl_total, 2)

    def get_realized_pnl_pct(self, market_snapshot: dict[str, float]) -> float:
        if self._realized_cost_total == 0:
            return 0.0
        realized_pnl = self.get_total_realized_pnl(market_snapshot)
        return round(realized_pnl / self._realized_cost_total, 4)


    def get_unrealized_pnl_pct(self, market_snapshot: dict[str, float]) -> float:
        total_cost = 0.0
        for asset_name, units in self.positions.items():
            if units <= 0 or asset_name not in market_snapshot:
                continue
            for buy_price, lot_units in self._get_open_lots_for_position(asset_name, units):
                total_cost += buy_price * lot_units
        if total_cost == 0:
            return 0.0
        unrealized_pnl = self.get_total_unrealized_pnl(market_snapshot)
        return round(unrealized_pnl / total_cost, 4)
    
    def deposit(self, ammount: float) -> None:
        self.deposit_history += [ammount]
        self.balance += ammount
        self.available_cash += ammount

    def liquidate(self, market_snapshot: dict[str, float]) -> None:
        missing_assets = [
            asset_name
            for asset_name, units in self.positions.items()
            if units > 0 and asset_name not in market_snapshot
        ]
        if missing_assets:
            raise ValueError(
                f"Cannot liquidate positions without market prices for: {', '.join(missing_assets)}"
            )

        for asset_name, units in list(self.positions.items()):
            if units <= 0:
                continue
            self.execute(asset_name=asset_name, units=-units, price=market_snapshot[asset_name])
        
    
