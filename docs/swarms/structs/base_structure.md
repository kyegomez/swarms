# Base Structure

`base_structure` reference documentation.

**Module Path**: `swarms.structs.base_structure`

## Overview

Base utility class that standardizes metadata, artifacts, errors, async helpers, and batching helpers for swarm structures.

## Public API

- **`BaseStructure`**: `run()`, `save_to_file()`, `load_from_file()`, `save_metadata()`, `load_metadata()`, `log_error()`, `save_artifact()`, `load_artifact()`

## Quickstart

```python
from swarms.structs.base_structure import BaseStructure
```

## Tutorial

A runnable tutorial is available at [`swarms/examples/base_structure_example.md`](../examples/base_structure_example.md).

## Notes

- Keep task payloads small for first runs.
- Prefer deterministic prompts when comparing outputs across agents.
- Validate provider credentials (for LLM-backed examples) before production use.
