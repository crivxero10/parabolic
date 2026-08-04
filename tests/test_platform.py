import tempfile
import unittest
from pathlib import Path

from parabolic.platform import RunStore, run_once, validate_cli_argv


ROOT = Path(__file__).resolve().parents[1]


class TestRunPlatform(unittest.TestCase):
    def test_rejects_unknown_commands_and_inline_secrets(self):
        with self.assertRaises(ValueError):
            validate_cli_argv(["unknown"])
        with self.assertRaises(ValueError):
            validate_cli_argv(["evaluate", "--api-key", "secret"])
        with self.assertRaises(ValueError):
            validate_cli_argv(["evaluate", "--api-secret=secret"])

    def test_worker_executes_existing_cli_and_records_evidence(self):
        with tempfile.TemporaryDirectory() as tempdir:
            state_dir = Path(tempdir)
            store = RunStore(state_dir)
            run = store.queue(["agent-spec"], None, ROOT)
            store.close()

            completed = run_once(state_dir, ROOT)

            self.assertIsNotNone(completed)
            self.assertEqual(completed.run_id, run.run_id)
            self.assertEqual(completed.status, "succeeded")
            self.assertEqual(completed.exit_code, 0)
            self.assertIn('"schema_version": 1', completed.stdout)
            self.assertEqual(completed.result["schema_version"], 1)

    def test_cancel_only_applies_to_queued_work(self):
        with tempfile.TemporaryDirectory() as tempdir:
            store = RunStore(Path(tempdir))
            run = store.queue(["agent-spec"], None, ROOT)
            cancelled = store.cancel(run.run_id)
            store.close()

            self.assertEqual(cancelled.status, "cancelled")

    def test_metadata_is_durable_and_editable(self):
        with tempfile.TemporaryDirectory() as tempdir:
            store = RunStore(Path(tempdir))
            run = store.queue(["agent-spec"], None, ROOT, {"model": "gpt-5", "effort": "high"})
            updated = store.update_metadata(run.run_id, {"model": "gpt-5", "effort": "medium"})
            store.close()

            self.assertEqual(updated.metadata, {"effort": "medium", "model": "gpt-5"})
