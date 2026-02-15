# Safe Loading

    `safe_loading` reference documentation.

    **Module Path**: `swarms.structs.safe_loading`

    ## Overview

    Safe state serialization/loading utilities that preserve object instances and only persist safe values.

    ## Public API

    - **`SafeLoaderUtils`**: `is_class_instance()`, `is_safe_type()`, `get_class_attributes()`, `create_state_dict()`, `preserve_instances()`
- **`SafeStateManager`**: `save_state()`, `load_state()`

    ## Quickstart

    ```python
    from swarms.structs.safe_loading import SafeLoaderUtils, SafeStateManager
    ```

    ## Tutorial

    A runnable tutorial is available at [`swarms/examples/safe_loading_example.md`](../examples/safe_loading_example.md).

    ## Notes

    - Keep task payloads small for first runs.
    - Prefer deterministic prompts when comparing outputs across agents.
    - Validate provider credentials (for LLM-backed examples) before production use.
