# Artifacts

The `artifacts` package provides structured file artifact management with version tracking, export helpers, and serialization utilities.

## Public API

From `swarms.artifacts`:

- `Artifact`

## Artifact Overview

`Artifact` is a Pydantic model that tracks file state over time:

- File metadata (`file_path`, `file_type`, `folder_path`)
- Current contents (`contents`)
- Edit lifecycle (`edit_count`, `versions`)
- Export and persistence helpers

`FileVersion` stores historical snapshots with `version_number`, `content`, and `timestamp`.

## Core Methods

- `create(initial_content)`: Initializes content and version history
- `edit(new_content)`: Appends a new version and increments edit count
- `save()` / `load()`: Persist or hydrate content from disk
- `get_version(version_number)`: Fetch a specific snapshot
- `get_version_history()`: Render all versions as text
- `export_to_json(file_path)` / `import_from_json(file_path)`
- `to_dict()` / `from_dict(data)`
- `save_as(output_format)`: Save as `.md`, `.txt`, `.pdf`, or `.py`

## Quickstart

```python
from swarms.artifacts import Artifact

artifact = Artifact(
    file_path="analysis.md",
    file_type=".md",
    contents="Initial draft",
    edit_count=0,
)

artifact.create("Initial draft")
artifact.edit("Revised draft with metrics")
artifact.save()

print(artifact.get_version_history())
```

## Notes

- Keep `file_type` aligned with `file_path` extension.
- Use `save_as(".pdf")` only where PDF dependencies are available.
- Use `export_to_json` for reproducible artifact state exchange.
