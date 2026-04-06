import math


class Indicators:

    @staticmethod
    def _validate_window(n: int) -> None:
        if n <= 0:
            raise ValueError("window must be positive")

    @staticmethod
    def _validate_equal_length(series_map: dict[str, list[float]]) -> None:
        lengths = {len(series) for series in series_map.values()}
        if len(lengths) > 1:
            raise ValueError("all input series must have the same length")

    @staticmethod
    def sma(series: list[float], n: int) -> list[float]:
        Indicators._validate_window(n)
        result = []
        for i in range(len(series)):
            if i + 1 < n:
                result.append(None)
            else:
                window = series[i+1-n:i+1]
                result.append(sum(window) / n)
        return result
    
    @staticmethod
    def ema(series: list[float], alpha: float) -> list[float]:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in the interval (0, 1]")
        result = []
        ema_prev = None
        for price in series:
            if ema_prev is None:
                ema_prev = price
            else:
                ema_prev = alpha * price + (1 - alpha) * ema_prev
            result.append(ema_prev)
        return result
    
    @staticmethod
    def ema_area_between_curves(
        series: list[float],
        k_st: int,
        k_lt: int,
        lookback: int,
    ) -> float:
        if k_st <= 0 or k_lt <= 0 or lookback <= 0:
            raise ValueError("k_st, k_lt, and lookback must be positive")
        if not series:
            raise ValueError("series cannot be empty")
        if len(series) < max(k_st, k_lt, lookback):
            raise ValueError("series does not have enough data for the requested parameters")

        alpha_st = 2 / (k_st + 1)
        alpha_lt = 2 / (k_lt + 1)

        st_curve = Indicators.ema(series, alpha_st)
        lt_curve = Indicators.ema(series, alpha_lt)

        diffs = [
            st - lt
            for st, lt in zip(st_curve[-lookback:], lt_curve[-lookback:])
        ]

        area = 0.0
        delta_x = 1.0
        for i in range(1, len(diffs)):
            area += delta_x * (diffs[i] + diffs[i - 1]) / 2

        return area
    

    @staticmethod
    def ema_window(n: int, series: list[float]) -> float:
        Indicators._validate_window(n)
        if len(series) < n:
            raise ValueError(f"Series does not have enough data to calculate {n} EMA")
        ema_slice = series[-n:]
        ema_result = ema_slice[0]
        alpha = 2 / (n + 1)
        for value in ema_slice[1:]:
            ema_result = alpha * value + (1 - alpha) * ema_result
        return ema_result

    @staticmethod
    def rolling_std(series: list[float], n: int) -> list[float | None]:
        Indicators._validate_window(n)
        result: list[float | None] = []
        for i in range(len(series)):
            if i + 1 < n:
                result.append(None)
                continue
            window = series[i + 1 - n : i + 1]
            mean_value = sum(window) / n
            variance = sum((value - mean_value) ** 2 for value in window) / n
            result.append(math.sqrt(variance))
        return result

    @staticmethod
    def bollinger_bands(
        series: list[float],
        n: int,
        num_std: float = 2.0,
    ) -> tuple[list[float | None], list[float | None], list[float | None]]:
        middle = Indicators.sma(series, n)
        rolling_std = Indicators.rolling_std(series, n)
        upper: list[float | None] = []
        lower: list[float | None] = []
        for mid, std in zip(middle, rolling_std):
            if mid is None or std is None:
                upper.append(None)
                lower.append(None)
                continue
            upper.append(mid + (num_std * std))
            lower.append(mid - (num_std * std))
        return lower, middle, upper

    @staticmethod
    def rsi(series: list[float], n: int = 14) -> list[float | None]:
        Indicators._validate_window(n)
        if not series:
            return []
        if len(series) == 1:
            return [None]

        result: list[float | None] = [None] * len(series)
        if len(series) <= n:
            return result

        gains: list[float] = []
        losses: list[float] = []
        for i in range(1, n + 1):
            delta = series[i] - series[i - 1]
            gains.append(max(delta, 0.0))
            losses.append(max(-delta, 0.0))

        avg_gain = sum(gains) / n
        avg_loss = sum(losses) / n
        result[n] = Indicators._rsi_from_averages(avg_gain, avg_loss)

        for i in range(n + 1, len(series)):
            delta = series[i] - series[i - 1]
            gain = max(delta, 0.0)
            loss = max(-delta, 0.0)
            avg_gain = ((avg_gain * (n - 1)) + gain) / n
            avg_loss = ((avg_loss * (n - 1)) + loss) / n
            result[i] = Indicators._rsi_from_averages(avg_gain, avg_loss)

        return result

    @staticmethod
    def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
        if avg_loss == 0 and avg_gain == 0:
            return 50.0
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def true_range(
        highs: list[float],
        lows: list[float],
        closes: list[float],
    ) -> list[float]:
        Indicators._validate_equal_length(
            {
                "highs": highs,
                "lows": lows,
                "closes": closes,
            }
        )
        if not highs:
            return []

        result = [highs[0] - lows[0]]
        for i in range(1, len(highs)):
            result.append(
                max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]),
                )
            )
        return result

    @staticmethod
    def atr(
        highs: list[float],
        lows: list[float],
        closes: list[float],
        n: int = 14,
    ) -> list[float | None]:
        Indicators._validate_window(n)
        tr = Indicators.true_range(highs, lows, closes)
        if not tr:
            return []

        result: list[float | None] = [None] * len(tr)
        if len(tr) < n:
            return result

        atr_value = sum(tr[:n]) / n
        result[n - 1] = atr_value
        for i in range(n, len(tr)):
            atr_value = ((atr_value * (n - 1)) + tr[i]) / n
            result[i] = atr_value
        return result

    @staticmethod
    def vwap(
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: list[float],
    ) -> list[float]:
        Indicators._validate_equal_length(
            {
                "highs": highs,
                "lows": lows,
                "closes": closes,
                "volumes": volumes,
            }
        )
        cumulative_price_volume = 0.0
        cumulative_volume = 0.0
        result: list[float] = []
        for high, low, close, volume in zip(highs, lows, closes, volumes):
            typical_price = (high + low + close) / 3.0
            cumulative_price_volume += typical_price * volume
            cumulative_volume += volume
            if cumulative_volume == 0:
                result.append(0.0)
            else:
                result.append(cumulative_price_volume / cumulative_volume)
        return result

    @staticmethod
    def macd(
        series: list[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[list[float], list[float], list[float]]:
        Indicators._validate_window(fast)
        Indicators._validate_window(slow)
        Indicators._validate_window(signal)
        if fast >= slow:
            raise ValueError("fast window must be smaller than slow window")
        if not series:
            return [], [], []

        fast_alpha = 2 / (fast + 1)
        slow_alpha = 2 / (slow + 1)
        signal_alpha = 2 / (signal + 1)

        fast_ema = Indicators.ema(series, fast_alpha)
        slow_ema = Indicators.ema(series, slow_alpha)
        macd_line = [fast_value - slow_value for fast_value, slow_value in zip(fast_ema, slow_ema)]
        signal_line = Indicators.ema(macd_line, signal_alpha)
        histogram = [macd_value - signal_value for macd_value, signal_value in zip(macd_line, signal_line)]
        return macd_line, signal_line, histogram

    @staticmethod
    def stochastic_oscillator(
        highs: list[float],
        lows: list[float],
        closes: list[float],
        k_period: int = 14,
        d_period: int = 3,
    ) -> tuple[list[float | None], list[float | None]]:
        Indicators._validate_window(k_period)
        Indicators._validate_window(d_period)
        Indicators._validate_equal_length(
            {
                "highs": highs,
                "lows": lows,
                "closes": closes,
            }
        )

        percent_k: list[float | None] = []
        for index in range(len(closes)):
            if index + 1 < k_period:
                percent_k.append(None)
                continue
            window_high = max(highs[index + 1 - k_period : index + 1])
            window_low = min(lows[index + 1 - k_period : index + 1])
            if window_high == window_low:
                percent_k.append(0.0)
            else:
                percent_k.append(
                    100.0 * ((closes[index] - window_low) / (window_high - window_low))
                )

        percent_d: list[float | None] = []
        for index in range(len(percent_k)):
            window = [value for value in percent_k[max(0, index + 1 - d_period) : index + 1] if value is not None]
            if len(window) < d_period:
                percent_d.append(None)
            else:
                percent_d.append(sum(window) / d_period)
        return percent_k, percent_d

    @staticmethod
    def williams_r(
        highs: list[float],
        lows: list[float],
        closes: list[float],
        n: int = 14,
    ) -> list[float | None]:
        Indicators._validate_window(n)
        Indicators._validate_equal_length(
            {
                "highs": highs,
                "lows": lows,
                "closes": closes,
            }
        )

        result: list[float | None] = []
        for index in range(len(closes)):
            if index + 1 < n:
                result.append(None)
                continue
            highest_high = max(highs[index + 1 - n : index + 1])
            lowest_low = min(lows[index + 1 - n : index + 1])
            if highest_high == lowest_low:
                result.append(0.0)
            else:
                result.append(
                    -100.0 * ((highest_high - closes[index]) / (highest_high - lowest_low))
                )
        return result

    @staticmethod
    def cci(
        highs: list[float],
        lows: list[float],
        closes: list[float],
        n: int = 20,
    ) -> list[float | None]:
        Indicators._validate_window(n)
        Indicators._validate_equal_length(
            {
                "highs": highs,
                "lows": lows,
                "closes": closes,
            }
        )

        typical_prices = [
            (high + low + close) / 3.0
            for high, low, close in zip(highs, lows, closes)
        ]
        sma_tp = Indicators.sma(typical_prices, n)
        result: list[float | None] = []
        for index, typical_price in enumerate(typical_prices):
            moving_average = sma_tp[index]
            if moving_average is None:
                result.append(None)
                continue
            window = typical_prices[index + 1 - n : index + 1]
            mean_deviation = sum(abs(value - moving_average) for value in window) / n
            if mean_deviation == 0:
                result.append(0.0)
            else:
                result.append((typical_price - moving_average) / (0.015 * mean_deviation))
        return result

    @staticmethod
    def mfi(
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: list[float],
        n: int = 14,
    ) -> list[float | None]:
        Indicators._validate_window(n)
        Indicators._validate_equal_length(
            {
                "highs": highs,
                "lows": lows,
                "closes": closes,
                "volumes": volumes,
            }
        )
        typical_prices = [
            (high + low + close) / 3.0
            for high, low, close in zip(highs, lows, closes)
        ]
        raw_money_flows = [
            typical_price * volume
            for typical_price, volume in zip(typical_prices, volumes)
        ]

        result: list[float | None] = [None] * len(closes)
        for index in range(n, len(closes)):
            positive_flow = 0.0
            negative_flow = 0.0
            for flow_index in range(index + 1 - n, index + 1):
                if flow_index == 0:
                    continue
                if typical_prices[flow_index] > typical_prices[flow_index - 1]:
                    positive_flow += raw_money_flows[flow_index]
                elif typical_prices[flow_index] < typical_prices[flow_index - 1]:
                    negative_flow += raw_money_flows[flow_index]
            if negative_flow == 0 and positive_flow == 0:
                result[index] = 50.0
            elif negative_flow == 0:
                result[index] = 100.0
            else:
                money_ratio = positive_flow / negative_flow
                result[index] = 100.0 - (100.0 / (1.0 + money_ratio))
        return result

    @staticmethod
    def obv(closes: list[float], volumes: list[float]) -> list[float]:
        Indicators._validate_equal_length(
            {
                "closes": closes,
                "volumes": volumes,
            }
        )
        if not closes:
            return []

        result = [float(volumes[0])]
        for index in range(1, len(closes)):
            if closes[index] > closes[index - 1]:
                result.append(result[-1] + float(volumes[index]))
            elif closes[index] < closes[index - 1]:
                result.append(result[-1] - float(volumes[index]))
            else:
                result.append(result[-1])
        return result
