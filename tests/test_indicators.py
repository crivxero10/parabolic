import unittest
from typing import Callable
import math
from parabolic.indicators import Indicators as ix

class TestIndicators(unittest.TestCase):
    
    def test_ema_known_values(self):

        data = [10, 20, 30]
        alpha = 0.5
        result = ix.ema(data, alpha)

        assert result[0] == 10
        assert math.isclose(result[1], 15.0)
        assert math.isclose(result[2], 22.5)

    def test_sma_known_values(self):

        data = [10, 20, 30, 40, 50]
        result = ix.sma(data, 3)
        expected = [None, None, 20.0, 30.0, 40.0]
        assert result == expected

    def test_ema_converges_on_constant_series(self):
        data = [5, 5, 5, 5, 5]
        result = ix.ema(data, 0.1)

        for value in result:
            assert math.isclose(value, 5.0)
