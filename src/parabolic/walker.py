import random
import secrets
from collections.abc import Sequence


class Walker:
    """Generates random walks."""

    def __init__(self, seed: int | float | str | bytes | bytearray | None = None):
        if seed is None:
            seed = secrets.randbits(256)
        self._rng = random.Random(seed)
        self.seed = seed

    def choice(self, items: Sequence[float] | Sequence[str] | Sequence[object]):
        if not items:
            raise ValueError("items must contain at least one value")
        return self._rng.choice(list(items))

    def weighted_choice(
        self,
        items: Sequence[float] | Sequence[str] | Sequence[object],
        weights: Sequence[float],
    ):
        if not items:
            raise ValueError("items must contain at least one value")
        if len(items) != len(weights):
            raise ValueError("items and weights must be the same length")
        if any(weight < 0 for weight in weights):
            raise ValueError("weights cannot be negative")
        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("weights must sum to a positive value")
        return self._rng.choices(list(items), weights=list(weights), k=1)[0]

    def random(self) -> float:
        return self._rng.random()

    def uniform(self, a: float, b: float) -> float:
        return self._rng.uniform(a, b)

    def gauss(self, mu: float, sigma: float) -> float:
        return self._rng.gauss(mu, sigma)

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
