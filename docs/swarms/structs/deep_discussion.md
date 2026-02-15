# Deep Discussion

`deep_discussion` reference documentation.

**Module Path**: `swarms.structs.deep_discussion`

## Overview

Two-agent turn-based debate primitive with conversation history formatting.

## Public API

- **`one_on_one_debate()`**

## Quickstart

```python
from swarms.structs.deep_discussion import one_on_one_debate
```

## Tutorial

A runnable tutorial is available at [`swarms/examples/deep_discussion_example.md`](../examples/deep_discussion_example.md).

## Notes

- Keep task payloads small for first runs.
- Prefer deterministic prompts when comparing outputs across agents.
- Validate provider credentials (for LLM-backed examples) before production use.
