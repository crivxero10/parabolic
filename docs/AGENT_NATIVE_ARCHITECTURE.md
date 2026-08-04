# Agent-Native Architecture

Parabolic remains a trading/backtesting CLI. This layer makes that system easier
to change safely by giving people and agents durable, machine-readable context
without adding a second application runtime.

## Source of Truth and Boundaries

The repository is the executable source of truth. [architecture/manifest.json](/Users/crivero/Documents/parabolic/parabolic/architecture/manifest.json)
declares each component's source paths, dependencies, ownership, and test paths.
Treat a component boundary as exclusive while a task is claimed. Interface changes
require an explicit contract task before dependent implementation work starts.

## Durable Work and Knowledge

Start each executable change from [architecture/TASK_TEMPLATE.md](/Users/crivero/Documents/parabolic/parabolic/architecture/TASK_TEMPLATE.md).
Record the Git snapshot, owned files, constraints, acceptance criteria, commands,
and evidence. Store architectural facts in [architecture/knowledge.json](/Users/crivero/Documents/parabolic/parabolic/architecture/knowledge.json), including provenance,
confidence, validity revision, and affected components. Facts whose source paths
or validity revision change must be reviewed or replaced.

## Deterministic Context Packets

`parabolic.agent_native.build_task_packet` is a pure function that normalizes a
task's scope into a bounded packet. It deliberately does not select files from
the filesystem, call a model, or mutate task state. Those are adapter concerns.
The companion `build_verification_evidence` produces a portable hash of a command
result, so task systems can correlate evidence without treating terminal output as
durable truth.

Inspect the machine-readable contract with:

```bash
uv run python -m parabolic.driver agent-spec
```

## Required Quality Gates

Every meaningful change must:

1. Name the component(s) and files it owns.
2. Preserve declared public contracts or update them deliberately.
3. Define acceptance criteria and reproducible verification commands.
4. Record verification evidence against the current Git snapshot.
5. Update or invalidate relevant knowledge facts.

The architecture test checks that declared component and test paths exist and that
knowledge facts reference known components. This turns the manifest into an
enforced contract rather than a one-time diagram.
