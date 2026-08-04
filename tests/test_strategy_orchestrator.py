import unittest
from datetime import date

from parabolic.strategy_orchestrator import Campaign, codex_home, compile_prompt, consistency_score, random_window


class TestStrategyOrchestrator(unittest.TestCase):
    def test_random_window_is_bounded_to_three_months(self):
        start, end = random_window(date(2026, 8, 4))
        self.assertLessEqual((date.fromisoformat(end) - date.fromisoformat(start)).days, 90)
        self.assertGreaterEqual((date.fromisoformat(end) - date.fromisoformat(start)).days, 20)

    def test_consistency_score_penalizes_drawdown(self):
        stable = {"sharpe": 1, "sortino": 2, "calmar": 2, "max_drawdown": -0.1}
        unstable = {"sharpe": 1, "sortino": 2, "calmar": 2, "max_drawdown": -2}
        self.assertGreater(consistency_score(stable), consistency_score(unstable))

    def test_prompt_is_bounded_and_records_selection(self):
        campaign = Campaign("c1", "SPY", "model", "active", 0)
        prompt, report = compile_prompt(campaign, None, [{"run_id": str(index), "result": {}, "metadata": {}} for index in range(10)])
        self.assertIn("SPY", prompt)
        self.assertIn("AUTHORITATIVE STRATEGY RUNTIME CONTRACT", prompt)
        self.assertIn("def strategy(ctx)", prompt)
        self.assertEqual(report["selected_result_count"], 6)
        self.assertEqual(len(report["strategy_contract_sha256"]), 64)

    def test_codex_provider_config_uses_environment_key(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tempdir:
            config = codex_home(__import__("pathlib").Path(tempdir)) / "config.toml"
            contents = config.read_text()
        self.assertIn('env_key = "OPENROUTER_API_KEY"', contents)
        self.assertIn('wire_api = "responses"', contents)
        self.assertNotIn("sk-", contents)
