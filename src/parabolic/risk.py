from __future__ import annotations
from math import sqrt
from typing import Iterable, Sequence

TRADING_DAYS_PER_YEAR = 252

def _to_float_list(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]


def _validate_non_empty(values: Sequence[float], name: str) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")


def _mean(values: Sequence[float]) -> float:
    _validate_non_empty(values, "values")
    return sum(values) / len(values)


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return sqrt(variance)


def equity_curve_from_returns(
    returns: Iterable[float],
    initial_equity: float = 1.0,
) -> list[float]:
    if initial_equity <= 0:
        raise ValueError("initial_equity must be positive")

    curve = [float(initial_equity)]
    equity = float(initial_equity)
    for period_return in _to_float_list(returns):
        equity *= 1.0 + period_return
        curve.append(equity)
    return curve


def returns_from_equity_curve(equity_curve: Iterable[float]) -> list[float]:
    curve = _to_float_list(equity_curve)
    if len(curve) < 2:
        return []

    returns: list[float] = []
    for previous, current in zip(curve[:-1], curve[1:]):
        if previous == 0:
            raise ValueError("equity curve contains a zero value, cannot compute returns")
        returns.append((current / previous) - 1.0)
    return returns


def max_drawdown(equity_curve: Iterable[float]) -> float:
    curve = _to_float_list(equity_curve)
    _validate_non_empty(curve, "equity_curve")

    peak = curve[0]
    worst_drawdown = 0.0
    for equity in curve:
        if equity > peak:
            peak = equity
        if peak == 0:
            continue
        drawdown = (equity / peak) - 1.0
        if drawdown < worst_drawdown:
            worst_drawdown = drawdown
    return worst_drawdown


def sharpe_ratio(
    returns: Iterable[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    values = _to_float_list(returns)
    _validate_non_empty(values, "returns")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    periodic_rf = risk_free_rate / periods_per_year
    excess_returns = [value - periodic_rf for value in values]
    volatility = _sample_std(excess_returns)
    if volatility == 0:
        return 0.0
    return (_mean(excess_returns) / volatility) * sqrt(periods_per_year)


def sortino_ratio(
    returns: Iterable[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    target_return: float | None = None,
) -> float:
    values = _to_float_list(returns)
    _validate_non_empty(values, "returns")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    periodic_target = (risk_free_rate / periods_per_year) if target_return is None else float(target_return)
    excess_returns = [value - periodic_target for value in values]
    downside_returns = [min(0.0, value) for value in excess_returns]
    downside_deviation = _sample_std(downside_returns)
    if downside_deviation == 0:
        return 0.0
    return (_mean(excess_returns) / downside_deviation) * sqrt(periods_per_year)


def calmar_ratio(
    returns: Iterable[float],
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    equity_curve: Iterable[float] | None = None,
) -> float:
    values = _to_float_list(returns)
    _validate_non_empty(values, "returns")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    if equity_curve is None:
        curve = equity_curve_from_returns(values)
    else:
        curve = _to_float_list(equity_curve)
        _validate_non_empty(curve, "equity_curve")

    worst_drawdown = abs(max_drawdown(curve))
    if worst_drawdown == 0:
        return 0.0

    compounded_growth = 1.0
    for period_return in values:
        compounded_growth *= 1.0 + period_return

    years = len(values) / periods_per_year
    if years <= 0:
        return 0.0

    annualized_return = compounded_growth ** (1.0 / years) - 1.0
    return annualized_return / worst_drawdown


def summarize_risk_metrics(
    returns: Iterable[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    equity_curve: Iterable[float] | None = None,
) -> dict[str, float]:
    values = _to_float_list(returns)
    _validate_non_empty(values, "returns")

    curve = equity_curve_from_returns(values) if equity_curve is None else _to_float_list(equity_curve)
    return {
        "sharpe": sharpe_ratio(values, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year),
        "sortino": sortino_ratio(values, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year),
        "calmar": calmar_ratio(values, periods_per_year=periods_per_year, equity_curve=curve),
        "max_drawdown": max_drawdown(curve),
    }