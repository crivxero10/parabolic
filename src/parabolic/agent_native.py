"""Deterministic contracts for agent-facing project work.

This module deliberately has no dependency on the CLI, model providers, or
filesystem.  Adapters may use these pure functions to build small, stable task
packets and verification evidence around the trading application.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class RepositorySnapshot:
    """Immutable repository state used to scope a task packet."""

    revision: str
    dirty: bool


@dataclass(frozen=True, slots=True)
class VerificationEvidence:
    """A reproducible result for one verification command."""

    command: str
    exit_code: int
    duration_ms: int
    output_sha256: str


def _stable_unique(values: Iterable[str]) -> list[str]:
    return sorted({value.strip() for value in values if value.strip()})


def build_task_packet(
    *,
    task_id: str,
    objective: str,
    snapshot: RepositorySnapshot,
    components: Sequence[str],
    files: Sequence[str],
    acceptance_criteria: Sequence[str],
    verification_commands: Sequence[str],
    constraints: Sequence[str] = (),
) -> dict[str, object]:
    """Build a deterministic, bounded task packet for a future agent run."""
    if not task_id.strip():
        raise ValueError("task_id must not be empty")
    if not objective.strip():
        raise ValueError("objective must not be empty")
    if not acceptance_criteria:
        raise ValueError("at least one acceptance criterion is required")
    if not verification_commands:
        raise ValueError("at least one verification command is required")

    return {
        "task_id": task_id.strip(),
        "objective": objective.strip(),
        "snapshot": asdict(snapshot),
        "components": _stable_unique(components),
        "files": _stable_unique(files),
        "acceptance_criteria": _stable_unique(acceptance_criteria),
        "verification_commands": _stable_unique(verification_commands),
        "constraints": _stable_unique(constraints),
    }


def build_verification_evidence(
    *, command: str, exit_code: int, duration_ms: int, output: str
) -> VerificationEvidence:
    """Create portable evidence without storing potentially noisy command output."""
    if not command.strip():
        raise ValueError("command must not be empty")
    if duration_ms < 0:
        raise ValueError("duration_ms must not be negative")
    return VerificationEvidence(
        command=command.strip(),
        exit_code=exit_code,
        duration_ms=duration_ms,
        output_sha256=sha256(output.encode("utf-8")).hexdigest(),
    )


def project_spec() -> dict[str, object]:
    """Machine-readable contract for the repository's agent-native scaffolding."""
    return {
        "schema_version": 1,
        "manifest": "architecture/manifest.json",
        "knowledge_store": "architecture/knowledge.json",
        "task_template": "architecture/TASK_TEMPLATE.md",
        "protocols": {
            "snapshot": ["revision", "dirty"],
            "task_packet": [
                "task_id",
                "objective",
                "snapshot",
                "components",
                "files",
                "acceptance_criteria",
                "verification_commands",
                "constraints",
            ],
            "verification_evidence": list(VerificationEvidence.__dataclass_fields__),
        },
        "quality_gates": [
            "Every change names its owned components and files.",
            "Every task defines acceptance criteria and verification commands.",
            "Verification evidence is correlated to an immutable repository snapshot.",
            "Knowledge facts name provenance, confidence, validity, and affected components.",
        ],
    }
