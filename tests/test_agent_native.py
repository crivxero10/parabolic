import json
import unittest
from pathlib import Path

from parabolic.agent_native import (
    RepositorySnapshot,
    build_task_packet,
    build_verification_evidence,
    project_spec,
)
from parabolic.driver import main


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class TestAgentNativeContracts(unittest.TestCase):
    def test_task_packet_is_deterministic_and_bounded(self):
        packet = build_task_packet(
            task_id=" TASK-12 ",
            objective=" Refactor market-data boundary ",
            snapshot=RepositorySnapshot(revision="abc123", dirty=False),
            components=["market-data", "cli", "market-data"],
            files=[" src/parabolic/mdp.py ", "src/parabolic/driver.py"],
            acceptance_criteria=[" Tests pass ", "Tests pass"],
            verification_commands=["pytest -q"],
            constraints=[" Preserve CLI output "],
        )

        self.assertEqual(packet["task_id"], "TASK-12")
        self.assertEqual(packet["components"], ["cli", "market-data"])
        self.assertEqual(packet["acceptance_criteria"], ["Tests pass"])
        self.assertEqual(packet["snapshot"], {"revision": "abc123", "dirty": False})

    def test_task_packet_requires_quality_gates(self):
        with self.assertRaises(ValueError):
            build_task_packet(
                task_id="TASK-13",
                objective="missing checks",
                snapshot=RepositorySnapshot(revision="abc123", dirty=False),
                components=[],
                files=[],
                acceptance_criteria=[],
                verification_commands=[],
            )

    def test_verification_evidence_is_reproducible(self):
        evidence = build_verification_evidence(
            command="pytest -q", exit_code=0, duration_ms=1250, output="114 passed\n"
        )

        self.assertEqual(evidence.command, "pytest -q")
        self.assertEqual(len(evidence.output_sha256), 64)
        self.assertEqual(evidence, build_verification_evidence(
            command="pytest -q", exit_code=0, duration_ms=1250, output="114 passed\n"
        ))

    def test_manifest_and_knowledge_are_well_formed(self):
        manifest = json.loads((REPOSITORY_ROOT / "architecture/manifest.json").read_text())
        knowledge = json.loads((REPOSITORY_ROOT / "architecture/knowledge.json").read_text())
        component_ids = {component["id"] for component in manifest["components"]}

        self.assertTrue(component_ids)
        for component in manifest["components"]:
            self.assertTrue(component["source_paths"])
            self.assertTrue(component["test_paths"])
            for path in [*component["source_paths"], *component["test_paths"]]:
                self.assertTrue((REPOSITORY_ROOT / path).exists(), path)
            self.assertTrue(set(component["depends_on"]).issubset(component_ids))

        for fact in knowledge["facts"]:
            self.assertTrue(set(fact["affected_components"]).issubset(component_ids))
            self.assertIn("provenance", fact)
            self.assertIn("confidence", fact)
            self.assertIn("valid_for", fact)

    def test_project_spec_exposes_required_protocols(self):
        spec = project_spec()

        self.assertEqual(spec["schema_version"], 1)
        self.assertIn("task_packet", spec["protocols"])
        self.assertIn("verification_evidence", spec["protocols"])

    def test_cli_emits_agent_spec_without_credentials(self):
        self.assertEqual(main(["agent-spec"]), 0)
