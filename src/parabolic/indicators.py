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
    
    