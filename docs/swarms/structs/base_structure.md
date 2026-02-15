# BaseStructure

    Reference documentation for `swarms.structs.base_structure`.

    ## Overview

    This module provides production utilities for `base structure` in Swarms.

    ## Module Path

    ```python
    from swarms.structs.base_structure import ...
    ```

    ## Public API

    - `BaseStructure`: `save_to_file`, `load_from_file`, `save_metadata`, `log_event`, async/thread helpers

    ## Quick Start

    ```python
    from swarms.structs.base_structure import BaseStructure

base = BaseStructure(name="demo", save_metadata_on=False)
base.save_to_file({"status": "ok"}, "./demo.json")
print(base.load_from_file("./demo.json"))
    ```

    ## Tutorial

    See the runnable tutorial: [`swarms/examples/base_structure_example.md`](../examples/base_structure_example.md)

    ## Operational Notes

    - Validate credentials and model access before running LLM-backed examples.
    - Start with small inputs/tasks, then scale once behavior is verified.
