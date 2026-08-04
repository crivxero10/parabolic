# Agent Task Template

Use one file or durable task-record per independently executable change. A task
must own a non-overlapping component, file set, or contract.

```yaml
id: TASK-###
objective: Concise outcome statement.
status: ready # ready | claimed | blocked | complete
snapshot: <git revision>
components: [component-id]
owned_files: [path]
must_not_change: [public contract or invariant]
dependencies: [TASK-###]
risks: [risk]
acceptance_criteria:
  - Observable condition.
verification_commands:
  - uv run --with pytest pytest -q
knowledge_updates:
  - fact-id or new fact description
evidence: []
```

Record claims, progress, blockers, and completion evidence in the task record.
Do not use conversation history as the only record of a decision or test result.
