class Indicators:
    
    @staticmethod
    def sma(series: list[float], n: int) -> list[float]:
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
        if len(series) < n:
            raise ValueError(f"Series does not have enough data to calculate {n} EMA")
        ema_slice = series[-n:]
        ema_result = ema_slice[0]
        alpha = 2 / (n + 1)
        for value in ema_slice[1:]:
            ema_result = alpha * value + (1 - alpha) * ema_result
        return ema_result