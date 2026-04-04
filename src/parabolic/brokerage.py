from dataclasses import dataclass

class Operation:
    def __init__(self, operation_type: str, asset_name: str, cost_basis: float):
        self.operation_type = operation_type
        self.asset_name = asset_name
        self.cost_basis = cost_basis

    def __str__(self):
        return f"{self.operation_type} {self.asset_name} @ {self.cost_basis}"
    
    def __repr__(self):
        return f"{self.operation_type} {self.asset_name} @ {self.cost_basis}"   

class Brokerage:
    def __init__(self, balance: float, positions: dict[str, int] | None = None, operations: list[Operation] | None = None):
        self.balance = balance
        self.positions = positions or {}
        self.operations = operations or []
    
    def execute(self, asset_name: str, units: int, price: float) -> bool:
        # BUY
        if units > 0:
            total_cost = units * price
            if total_cost > self.balance:
                return False
            self.balance -= total_cost

            if asset_name not in self.positions:
                self.positions[asset_name] = 0
            self.positions[asset_name] += units

            for _ in range(units):
                self.operations.append(Operation("BUY", asset_name, price))

            return True

        # SELL
        if units < 0:
            sell_units = abs(units)

            # not enough position
            if asset_name not in self.positions or self.positions[asset_name] < sell_units:
                return False

            self.positions[asset_name] -= sell_units
            self.balance += sell_units * price

            for _ in range(sell_units):
                self.operations.append(Operation("SELL", asset_name, price))

            return True

        return False

    def get_total_unrealized_pnl(self, market_snapshot: dict[str, float]) -> float:
        # build cost basis per asset from operations
        pnl = 0.0
        cost_basis_map = {}
        for op in self.operations:
            if op.operation_type != "BUY":
                continue
            if op.asset_name not in cost_basis_map:
                cost_basis_map[op.asset_name] = []
            cost_basis_map[op.asset_name].append(op.cost_basis)

        for asset_name, units in self.positions.items():
            if asset_name not in market_snapshot:
                continue

            market_price = market_snapshot[asset_name]

            # if no operations recorded, assume default cost basis = 0
            cost_list = cost_basis_map.get(asset_name, [])
            if cost_list:
                avg_cost = sum(cost_list) / len(cost_list)
            else:
                avg_cost = 0.0
            
            # derive cost basis from BUY operations
            cost_list = [
                op.cost_basis
                for op in self.operations
                if op.operation_type == "BUY" and op.asset_name == asset_name
            ]

            pnl += units * (market_price - avg_cost)

        return round(pnl, 2)
    
    def get_total_realized_pnl(self, market_snapshot: dict[str, float]) -> float:
        pnl = 0.0
        # FIFO inventory per asset
        inventory: dict[str, list[float]] = {}

        for op in self.operations:
            asset = op.asset_name

            if asset not in inventory:
                inventory[asset] = []

            if op.operation_type == "BUY":
                inventory[asset].append(op.cost_basis)

            elif op.operation_type == "SELL":
                # only realized if there is inventory to match against
                if not inventory[asset]:
                    continue

                buy_cost = inventory[asset].pop(0)
                pnl += op.cost_basis - buy_cost

        return round(pnl, 2)

