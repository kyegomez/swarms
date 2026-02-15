# Agent Grpo

`agent_grpo` reference documentation.

**Module Path**: `swarms.structs.agent_grpo`

## Overview

Group-relative policy optimization utility for sampling, grading, and advantage computation across multiple agent outputs.

## Public API

- **`AgenticGRPO`**: `sample()`, `rate_answers_to_correct_answer()`, `rate_answer_to_correct_answer()`, `compute_group_baseline()`, `compute_advantages()`, `get_correct_responses()`, `run()`, `get_all()`

## Quickstart

```python
from swarms.structs.agent_grpo import AgenticGRPO
```

## Tutorial

A runnable tutorial is available at [`swarms/examples/agent_grpo_example.md`](../examples/agent_grpo_example.md).

## Notes

- Keep task payloads small for first runs.
- Prefer deterministic prompts when comparing outputs across agents.
- Validate provider credentials (for LLM-backed examples) before production use.
