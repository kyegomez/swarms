# Collaborative Utils

`collaborative_utils` reference documentation.

**Module Path**: `swarms.structs.collaborative_utils`

## Overview

Targeted inter-agent collaboration helper that routes one agent to a named peer using one-on-one debate.

## Public API

- **`talk_to_agent()`**

## Quickstart

```python
from swarms.structs.collaborative_utils import talk_to_agent
```

## Tutorial

A runnable tutorial is available at [`swarms/examples/collaborative_utils_example.md`](../examples/collaborative_utils_example.md).

## Notes

- Keep task payloads small for first runs.
- Prefer deterministic prompts when comparing outputs across agents.
- Validate provider credentials (for LLM-backed examples) before production use.
