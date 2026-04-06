import math
import unittest
from parabolic.risk import calmar_ratio, sharpe_ratio, sortino_ratio

class TestRisk(unittest.TestCase):
    def test_sharpe_ratio(self):
        returns = [0.01, 0.02, -0.01, 0.015, 0.005]

        mean_return = sum(returns) / len(returns)
        variance = sum((value - mean_return) ** 2 for value in returns) / (len(returns) - 1)
        std_dev = math.sqrt(variance)
        expected = (mean_return / std_dev) * math.sqrt(252)

        result = sharpe_ratio(returns)

        assert round(result, 10) == round(expected, 10)

    def test_sortino_ratio(self):
        returns = [0.01, 0.02, -0.01, 0.015, 0.005]

        excess_returns = returns
        downside_returns = [min(0.0, value) for value in excess_returns]
        mean_return = sum(excess_returns) / len(excess_returns)
        downside_mean = sum(downside_returns) / len(downside_returns)
        downside_variance = sum((value - downside_mean) ** 2 for value in downside_returns) / (len(downside_returns) - 1)
        downside_std = math.sqrt(downside_variance)
        expected = (mean_return / downside_std) * math.sqrt(252)

        result = sortino_ratio(returns)

        assert round(result, 10) == round(expected, 10)

    def test_calmar_ratio(self):
        returns = [0.10, -0.05, 0.03, -0.02, 0.04]

        compounded_growth = 1.0
        for period_return in returns:
            compounded_growth *= 1.0 + period_return
        years = len(returns) / 252
        annualized_return = compounded_growth ** (1.0 / years) - 1.0

        equity_curve = [1.0]
        equity = 1.0
        for period_return in returns:
            equity *= 1.0 + period_return
            equity_curve.append(equity)

        peak = equity_curve[0]
        worst_drawdown = 0.0
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (value / peak) - 1.0
            if drawdown < worst_drawdown:
                worst_drawdown = drawdown

        expected = annualized_return / abs(worst_drawdown)

        result = calmar_ratio(returns)

        assert round(result, 10) == round(expected, 10)