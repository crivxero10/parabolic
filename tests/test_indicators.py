import unittest
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

    def test_rolling_std_known_values(self):
        data = [1.0, 2.0, 3.0, 4.0]

        result = ix.rolling_std(data, 2)

        expected_std = math.sqrt(0.25)
        assert result[0] is None
        assert math.isclose(result[1], expected_std)
        assert math.isclose(result[2], expected_std)
        assert math.isclose(result[3], expected_std)

    def test_bollinger_bands_known_values(self):
        data = [1.0, 2.0, 3.0]

        lower, middle, upper = ix.bollinger_bands(data, 3, num_std=2.0)

        std = math.sqrt(2.0 / 3.0)
        assert lower == [None, None, 2.0 - (2.0 * std)]
        assert middle == [None, None, 2.0]
        assert upper == [None, None, 2.0 + (2.0 * std)]

    def test_rsi_rises_to_100_on_monotonic_gains(self):
        data = [1.0, 2.0, 3.0, 4.0]

        result = ix.rsi(data, 2)

        assert result[:2] == [None, None]
        assert math.isclose(result[2], 100.0)
        assert math.isclose(result[3], 100.0)

    def test_rsi_is_50_on_flat_series(self):
        data = [5.0, 5.0, 5.0, 5.0]

        result = ix.rsi(data, 2)

        assert result[:2] == [None, None]
        assert math.isclose(result[2], 50.0)
        assert math.isclose(result[3], 50.0)

    def test_true_range_known_values(self):
        highs = [10.0, 11.0, 12.0]
        lows = [9.0, 9.5, 10.0]
        closes = [9.5, 10.0, 11.0]

        result = ix.true_range(highs, lows, closes)

        assert result == [1.0, 1.5, 2.0]

    def test_atr_known_values(self):
        highs = [10.0, 11.0, 12.0]
        lows = [9.0, 9.5, 10.0]
        closes = [9.5, 10.0, 11.0]

        result = ix.atr(highs, lows, closes, 2)

        assert result[0] is None
        assert math.isclose(result[1], 1.25)
        assert math.isclose(result[2], 1.625)

    def test_vwap_known_values(self):
        highs = [10.0, 20.0]
        lows = [8.0, 18.0]
        closes = [9.0, 19.0]
        volumes = [100.0, 300.0]

        result = ix.vwap(highs, lows, closes, volumes)

        assert math.isclose(result[0], 9.0)
        assert math.isclose(result[1], 16.5)

    def test_macd_is_zero_on_constant_series(self):
        data = [5.0] * 10

        macd_line, signal_line, histogram = ix.macd(data, fast=3, slow=6, signal=3)

        assert macd_line == [0.0] * 10
        assert signal_line == [0.0] * 10
        assert histogram == [0.0] * 10

    def test_macd_turns_positive_on_rising_series(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        macd_line, signal_line, histogram = ix.macd(data, fast=2, slow=4, signal=2)

        assert len(macd_line) == len(data)
        assert len(signal_line) == len(data)
        assert len(histogram) == len(data)
        assert macd_line[-1] > 0.0
        assert signal_line[-1] > 0.0

    def test_stochastic_oscillator_known_values(self):
        highs = [10.0, 12.0, 14.0]
        lows = [5.0, 6.0, 7.0]
        closes = [9.0, 11.0, 13.0]

        percent_k, percent_d = ix.stochastic_oscillator(highs, lows, closes, k_period=2, d_period=2)

        assert percent_k[0] is None
        assert math.isclose(percent_k[1], 85.71428571428571)
        assert math.isclose(percent_k[2], 87.5)
        assert percent_d[0] is None
        assert percent_d[1] is None
        assert math.isclose(percent_d[2], (85.71428571428571 + 87.5) / 2.0)

    def test_williams_r_known_values(self):
        highs = [10.0, 12.0, 14.0]
        lows = [5.0, 6.0, 7.0]
        closes = [9.0, 11.0, 13.0]

        result = ix.williams_r(highs, lows, closes, 2)

        assert result[0] is None
        assert math.isclose(result[1], -14.285714285714285)
        assert math.isclose(result[2], -12.5)

    def test_cci_known_values(self):
        highs = [10.0, 12.0, 14.0]
        lows = [8.0, 10.0, 12.0]
        closes = [9.0, 11.0, 13.0]

        result = ix.cci(highs, lows, closes, 3)

        assert result[:2] == [None, None]
        assert math.isclose(result[2], 100.0)

    def test_mfi_rises_to_100_on_consistent_positive_flow(self):
        highs = [10.0, 11.0, 12.0, 13.0]
        lows = [8.0, 9.0, 10.0, 11.0]
        closes = [9.0, 10.0, 11.0, 12.0]
        volumes = [100.0, 100.0, 100.0, 100.0]

        result = ix.mfi(highs, lows, closes, volumes, 2)

        assert result[:2] == [None, None]
        assert math.isclose(result[2], 100.0)
        assert math.isclose(result[3], 100.0)

    def test_obv_known_values(self):
        closes = [10.0, 11.0, 10.0, 10.0]
        volumes = [100.0, 150.0, 120.0, 130.0]

        result = ix.obv(closes, volumes)

        assert result == [100.0, 250.0, 130.0, 130.0]

    def test_indicator_validation_raises_for_invalid_inputs(self):
        with self.assertRaises(ValueError):
            ix.sma([1.0, 2.0], 0)
        with self.assertRaises(ValueError):
            ix.ema([1.0, 2.0], 0.0)
        with self.assertRaises(ValueError):
            ix.true_range([1.0], [1.0, 2.0], [1.0])
        with self.assertRaises(ValueError):
            ix.macd([1.0, 2.0], fast=5, slow=3, signal=2)
        with self.assertRaises(ValueError):
            ix.stochastic_oscillator([1.0], [1.0, 2.0], [1.0], 2, 2)
        with self.assertRaises(ValueError):
            ix.obv([1.0], [1.0, 2.0])
