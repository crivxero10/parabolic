from collections import deque
from itertools import count
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from parabolic.backtest import TradingContext

class Operation:

    def __init__(
        self,
        operation_type: str,
        asset_name: str,
        cost_basis: float,
        timestamp: str | None = None,
        units: int = 1,
        position_id: str | None = None,
    ):
        self.operation_type = operation_type
        self.asset_name = asset_name
        self.cost_basis = cost_basis
        self.timestamp = timestamp
        self.units = units
        self.position_id = position_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_type": self.operation_type,
            "asset_name": self.asset_name,
            "cost_basis": self.cost_basis,
            "timestamp": self.timestamp,
            "units": self.units,
            "position_id": self.position_id,
        }

    def __str__(self):
        suffix = f" ts={self.timestamp}" if self.timestamp is not None else ""
        position = f" position_id={self.position_id}" if self.position_id is not None else ""
        return f"{self.operation_type} {self.asset_name} {self.units} @ {self.cost_basis}{suffix}{position}"
    
    def __repr__(self):
        return self.__str__()
    
class Instruction:

    def __init__(
        self,
        asset_name: str,
        units: int,
        target_price: float,
        activate: Callable[..., bool],
        position_id: str | None = None,
    ):
        self.asset_name = asset_name
        self.units = units
        self.target_price = target_price
        self.activate = activate
        self.position_id = position_id

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
        self._position_id_counter = count(1)
        self._open_positions: dict[str, deque[dict[str, Any]]] = {}
        self._closed_trades: list[dict[str, Any]] = []
        self._execution_log: list[dict[str, Any]] = []
        self._rebuild_caches_from_operations()

    def _rebuild_caches_from_operations(self) -> None:
        self._inventory_lots = {}
        self._realized_pnl_total = 0.0
        self._realized_cost_total = 0.0
        self._open_positions = {}
        self._closed_trades = []
        self._execution_log = []
        self._position_id_counter = count(1)
        for op in self.operations:
            units = max(int(getattr(op, "units", 1)), 1)
            timestamp = getattr(op, "timestamp", None)
            position_id = getattr(op, "position_id", None)
            if op.operation_type == "BUY":
                self._record_buy(op.asset_name, op.cost_basis, units)
                position_ids = self._record_open_position(
                    asset_name=op.asset_name,
                    price=op.cost_basis,
                    units=units,
                    timestamp=timestamp,
                    position_id=position_id,
                )
                self._record_execution(
                    operation_type="BUY",
                    asset_name=op.asset_name,
                    price=op.cost_basis,
                    units=units,
                    timestamp=timestamp,
                    position_id=position_ids[0] if position_ids else position_id,
                    pnl_amount=0.0,
                    pnl_pct=0.0,
                )
            elif op.operation_type == "SELL":
                self._record_sell(op.asset_name, op.cost_basis, units)
                closed_trades = self._record_close_position(
                    asset_name=op.asset_name,
                    price=op.cost_basis,
                    units=units,
                    timestamp=timestamp,
                )
                if closed_trades:
                    for closed_trade in closed_trades:
                        self._closed_trades.append(closed_trade)
                        self._record_execution(
                            operation_type="SELL",
                            asset_name=op.asset_name,
                            price=op.cost_basis,
                            units=int(closed_trade["quantity"]),
                            timestamp=timestamp,
                            position_id=str(closed_trade["position_id"]),
                            pnl_amount=float(closed_trade["pnl_amount"]),
                            pnl_pct=float(closed_trade["pnl_pct"]),
                        )
                else:
                    self._record_execution(
                        operation_type="SELL",
                        asset_name=op.asset_name,
                        price=op.cost_basis,
                        units=units,
                        timestamp=timestamp,
                        position_id=position_id,
                        pnl_amount=0.0,
                        pnl_pct=0.0,
                    )

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

    def _next_position_id(self) -> str:
        return f"position-{next(self._position_id_counter)}"

    def _record_open_position(
        self,
        *,
        asset_name: str,
        price: float,
        units: int,
        timestamp: str | None,
        position_id: str | None = None,
    ) -> list[str]:
        created_position_ids: list[str] = []
        if units <= 0:
            return created_position_ids
        positions = self._open_positions.setdefault(asset_name, deque())
        current_position_id = position_id or self._next_position_id()
        created_position_ids.append(current_position_id)
        positions.append(
            {
                "position_id": current_position_id,
                "asset": asset_name,
                "side": "long",
                "entry_timestamp": timestamp,
                "entry_price": float(price),
                "quantity": int(units),
                "running_cost_basis": float(price) * int(units),
                "realized_pnl": 0.0,
                "entry_operation_type": "BUY",
                "bars_held": None,
                "fees": None,
                "slippage": None,
            }
        )
        return created_position_ids

    def _record_close_position(
        self,
        *,
        asset_name: str,
        price: float,
        units: int,
        timestamp: str | None,
    ) -> list[dict[str, Any]]:
        if units <= 0:
            return []
        positions = self._open_positions.setdefault(asset_name, deque())
        closed_trades: list[dict[str, Any]] = []
        remaining_units = int(units)
        while remaining_units > 0 and positions:
            open_position = positions.popleft()
            open_quantity = int(open_position["quantity"])
            matched_units = min(remaining_units, open_quantity)
            entry_price = float(open_position["entry_price"])
            total_cost_basis = float(open_position.get("running_cost_basis", entry_price * open_quantity))
            matched_cost_basis = total_cost_basis * (matched_units / open_quantity)
            pnl_amount = (float(price) - entry_price) * matched_units
            pnl_pct = 0.0 if matched_cost_basis == 0 else pnl_amount / matched_cost_basis
            closed_trade = {
                "position_id": str(open_position["position_id"]),
                "asset": asset_name,
                "side": str(open_position["side"]),
                "entry_timestamp": open_position["entry_timestamp"],
                "exit_timestamp": timestamp,
                "entry_price": entry_price,
                "exit_price": float(price),
                "quantity": matched_units,
                "running_cost_basis": matched_cost_basis,
                "realized_pnl": pnl_amount,
                "pnl_amount": pnl_amount,
                "pnl_pct": pnl_pct,
                "bars_held": open_position.get("bars_held"),
                "fees": open_position.get("fees"),
                "slippage": open_position.get("slippage"),
            }
            closed_trades.append(closed_trade)

            remaining_units -= matched_units
            remaining_quantity = open_quantity - matched_units
            if remaining_quantity > 0:
                remaining_position = dict(open_position)
                remaining_position["quantity"] = remaining_quantity
                remaining_position["running_cost_basis"] = total_cost_basis - matched_cost_basis
                positions.appendleft(remaining_position)
        return closed_trades

    def _record_execution(
        self,
        *,
        operation_type: str,
        asset_name: str,
        price: float,
        units: int,
        timestamp: str | None,
        position_id: str | None,
        pnl_amount: float,
        pnl_pct: float,
    ) -> None:
        self._execution_log.append(
            {
                "operation_type": operation_type,
                "asset": asset_name,
                "price": float(price),
                "units": int(units),
                "timestamp": timestamp,
                "position_id": position_id,
                "pnl_amount": float(pnl_amount),
                "pnl_pct": float(pnl_pct),
            }
        )

    def _expand_closed_trade_legacy(self, trade: dict[str, Any]) -> list[dict[str, Any]]:
        quantity = max(int(trade.get("quantity", 1)), 1)
        pnl_amount_total = float(trade.get("pnl_amount", 0.0))
        running_cost_basis_total = float(trade.get("running_cost_basis", 0.0))
        expanded: list[dict[str, Any]] = []
        for _ in range(quantity):
            expanded.append(
                {
                    "position_id": trade.get("position_id"),
                    "asset": trade.get("asset"),
                    "side": trade.get("side"),
                    "entry_timestamp": trade.get("entry_timestamp"),
                    "exit_timestamp": trade.get("exit_timestamp"),
                    "entry_price": trade.get("entry_price"),
                    "exit_price": trade.get("exit_price"),
                    "quantity": 1,
                    "running_cost_basis": running_cost_basis_total / quantity if quantity else 0.0,
                    "realized_pnl": pnl_amount_total / quantity if quantity else 0.0,
                    "pnl_amount": pnl_amount_total / quantity if quantity else 0.0,
                    "pnl_pct": trade.get("pnl_pct"),
                    "bars_held": trade.get("bars_held"),
                    "fees": trade.get("fees"),
                    "slippage": trade.get("slippage"),
                }
            )
        return expanded

    def _expand_execution_entry_legacy(self, entry: dict[str, Any]) -> list[dict[str, Any]]:
        units = max(int(entry.get("units", 1)), 1)
        pnl_amount_total = float(entry.get("pnl_amount", 0.0))
        expanded: list[dict[str, Any]] = []
        for _ in range(units):
            expanded.append(
                {
                    "operation_type": entry.get("operation_type"),
                    "asset": entry.get("asset"),
                    "price": float(entry.get("price", 0.0)),
                    "units": 1,
                    "timestamp": entry.get("timestamp"),
                    "position_id": entry.get("position_id"),
                    "pnl_amount": pnl_amount_total / units if units else 0.0,
                    "pnl_pct": float(entry.get("pnl_pct", 0.0)),
                }
            )
        return expanded

    def _append_operations(
        self,
        operation_type: str,
        asset_name: str,
        price: float,
        units: int,
        *,
        timestamp: str | None = None,
        position_ids: list[str] | None = None,
        position_units: list[int] | None = None,
    ) -> None:
        if units <= 0:
            return
        generated_operations: list[Operation] = []
        if position_ids and position_units and len(position_ids) == len(position_units):
            for current_position_id, current_units in zip(position_ids, position_units):
                for _ in range(max(int(current_units), 0)):
                    generated_operations.append(
                        Operation(
                            operation_type,
                            asset_name,
                            price,
                            timestamp=timestamp,
                            units=1,
                            position_id=current_position_id,
                        )
                    )
        elif position_ids and len(position_ids) == 1:
            for _ in range(int(units)):
                generated_operations.append(
                    Operation(
                        operation_type,
                        asset_name,
                        price,
                        timestamp=timestamp,
                        units=1,
                        position_id=position_ids[0],
                    )
                )
        else:
            for index in range(int(units)):
                generated_operations.append(
                    Operation(
                        operation_type,
                        asset_name,
                        price,
                        timestamp=timestamp,
                        units=1,
                        position_id=None if not position_ids or index >= len(position_ids) else position_ids[index],
                    )
                )
        self.operations.extend(generated_operations)

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

    def execute(self, asset_name: str, units: int, price: float, timestamp: str | None = None) -> bool:
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
            position_ids = self._record_open_position(
                asset_name=asset_name,
                price=price,
                units=units,
                timestamp=timestamp,
            )
            self._append_operations(
                "BUY",
                asset_name,
                price,
                units,
                timestamp=timestamp,
                position_ids=position_ids,
            )
            self._record_execution(
                operation_type="BUY",
                asset_name=asset_name,
                price=price,
                units=units,
                timestamp=timestamp,
                position_id=position_ids[0] if position_ids else None,
                pnl_amount=0.0,
                pnl_pct=0.0,
            )
            return True
        # SELL
        if units < 0:
            sell_units = abs(units)
            sale_proceeds = sell_units * price
            # not enough position
            if asset_name not in self.positions or self.positions[asset_name] < sell_units:
                return False
            closed_trades = self._record_close_position(
                asset_name=asset_name,
                price=price,
                units=sell_units,
                timestamp=timestamp,
            )
            self.positions[asset_name] -= sell_units
            self.balance += sale_proceeds
            if not self.settled_cash_only:
                self.available_cash += sale_proceeds
            self._record_sell(asset_name, price, sell_units)
            position_ids = [str(trade["position_id"]) for trade in closed_trades]
            position_units = [int(trade["quantity"]) for trade in closed_trades]
            self._append_operations(
                "SELL",
                asset_name,
                price,
                sell_units,
                timestamp=timestamp,
                position_ids=position_ids,
                position_units=position_units,
            )
            for closed_trade in closed_trades:
                self._closed_trades.append(closed_trade)
                self._record_execution(
                    operation_type="SELL",
                    asset_name=asset_name,
                    price=price,
                    units=int(closed_trade["quantity"]),
                    timestamp=timestamp,
                    position_id=str(closed_trade["position_id"]),
                    pnl_amount=float(closed_trade["pnl_amount"]),
                    pnl_pct=float(closed_trade["pnl_pct"]),
                )
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
                    timestamp=str(ctx.t),
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

    def get_closed_trades(self) -> list[dict[str, Any]]:
        expanded: list[dict[str, Any]] = []
        for trade in self._closed_trades:
            expanded.extend(self._expand_closed_trade_legacy(trade))
        return expanded

    def get_execution_log(self) -> list[dict[str, Any]]:
        expanded: list[dict[str, Any]] = []
        for entry in self._execution_log:
            expanded.extend(self._expand_execution_entry_legacy(entry))
        return expanded

    def get_order_log(self) -> list[dict[str, Any]]:
        return self.get_execution_log()
    
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
            units = max(int(getattr(op, "units", 1)), 1)
            if op.operation_type == "BUY":
                lots = inventory[asset]
                if lots and lots[-1][0] == op.cost_basis:
                    lots[-1][1] += units
                else:
                    lots.append([op.cost_basis, units])
            elif op.operation_type == "SELL":
                lots = inventory[asset]
                remaining_units = units
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
        
    
