import unittest
from parabolic.walker import Walker


class TestWalker(unittest.TestCase):

    def test_walk_extends_series_with_seeded_randomness(self):
        walker = Walker(seed=123)
        series = [100.0, 102.0, 101.0, 104.0]

        result = walker.walk(series=series, n=5)

        assert result[:len(series)] == series
        assert len(result) == len(series) + 5

        step_pool = {2.0, -1.0, 3.0}
        generated_steps = [result[i] - result[i - 1] for i in range(len(series), len(result))]
        assert all(step in step_pool for step in generated_steps)

        walker_again = Walker(seed=123)
        assert result == walker_again.walk(series=series, n=5)

    def test_walk_raises_on_empty_series(self):
        walker = Walker(seed=123)

        with self.assertRaises(ValueError):
            walker.walk(series=[], n=3)

    def test_walk_raises_on_negative_n(self):
        walker = Walker(seed=123)

        with self.assertRaises(ValueError):
            walker.walk(series=[100.0], n=-1)