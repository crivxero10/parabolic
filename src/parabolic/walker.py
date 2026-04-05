import random
import secrets


class Walker:
    """Generates random walks."""

    def __init__(self, seed: int | float | str | bytes | bytearray | None = None):
        if seed is None:
            seed = secrets.randbits(256)
        self._rng = random.Random(seed)
        self.seed = seed

    def walk(self, series: list[float], n: int) -> list[float]:
        if not series:
            raise ValueError("series must contain at least one value")
        if n < 0:
            raise ValueError("n must be non-negative")

        path = list(series)
        if len(series) == 1:
            step_pool = [0.0]
        else:
            step_pool = [series[i] - series[i - 1] for i in range(1, len(series))]

        current = path[-1]
        for _ in range(n):
            step = self._rng.choice(step_pool)
            current += step
            path.append(current)

        return path