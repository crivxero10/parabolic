from dataclasses import dataclass
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
            for _ in range(units):
                self.operations.append(Operation("BUY", asset_name, price))
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
            for _ in range(sell_units):
                self.operations.append(Operation("SELL", asset_name, price))
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
        for op in self.operations:
            asset = op.asset_name
            inventory.setdefault(asset, [])
            if op.operation_type == "BUY":
                inventory[asset].append(op.cost_basis)
            elif op.operation_type == "SELL":
                if inventory[asset]:
                    inventory[asset].pop(0)
        return inventory

    def _get_realized_matches(self) -> list[tuple[float, float]]:
        inventory: dict[str, list[float]] = {}
        matches: list[tuple[float, float]] = []
        for op in self.operations:
            asset = op.asset_name
            inventory.setdefault(asset, [])
            if op.operation_type == "BUY":
                inventory[asset].append(op.cost_basis)
            elif op.operation_type == "SELL":
                if not inventory[asset]:
                    continue
                buy_cost = inventory[asset].pop(0)
                matches.append((buy_cost, op.cost_basis))
        return matches

    def get_total_unrealized_pnl(self, market_snapshot: dict[str, float]) -> float:
        pnl = 0.0
        inventory = self._get_open_inventory()
        for asset_name, units in self.positions.items():
            if units <= 0 or asset_name not in market_snapshot:
                continue
            open_costs = inventory.get(asset_name, [])[:units]
            if not open_costs:
                continue
            avg_cost = sum(open_costs) / len(open_costs)
            pnl += len(open_costs) * (market_snapshot[asset_name] - avg_cost)
        return round(pnl, 2)
    
    def get_total_realized_pnl(self, market_snapshot: dict[str, float]) -> float:
        pnl = 0.0
        for buy_cost, sell_cost in self._get_realized_matches():
            pnl += sell_cost - buy_cost
        return round(pnl, 2)

    def get_realized_pnl_pct(self, market_snapshot: dict[str, float]) -> float:
        realized_cost = sum(buy_cost for buy_cost, _ in self._get_realized_matches())
        if realized_cost == 0:
            return 0.0
        realized_pnl = self.get_total_realized_pnl(market_snapshot)
        return round(realized_pnl / realized_cost, 4)


    def get_unrealized_pnl_pct(self, market_snapshot: dict[str, float]) -> float:
        inventory = self._get_open_inventory()
        total_cost = 0.0
        for asset_name, units in self.positions.items():
            if units <= 0 or asset_name not in market_snapshot:
                continue
            open_costs = inventory.get(asset_name, [])[:units]
            total_cost += sum(open_costs)
        if total_cost == 0:
            return 0.0
        unrealized_pnl = self.get_total_unrealized_pnl(market_snapshot)
        return round(unrealized_pnl / total_cost, 4)
