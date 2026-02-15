# Stopping Conditions

    Reference documentation for `swarms.structs.stopping_conditions`.

    ## Overview

    This module provides production utilities for `stopping conditions` in Swarms.

    ## Module Path

    ```python
    from swarms.structs.stopping_conditions import ...
    ```

    ## Public API

    - `check_stopping_conditions(input)` plus predicate helpers like `check_done`, `check_finished`

    ## Quick Start

    ```python
    from swarms.structs.stopping_conditions import check_stopping_conditions

print(check_stopping_conditions("task finished successfully"))
print(check_stopping_conditions("continue running"))
    ```

    ## Tutorial

    See the runnable tutorial: [`swarms/examples/stopping_conditions_example.md`](../examples/stopping_conditions_example.md)

    ## Operational Notes

    - Validate credentials and model access before running LLM-backed examples.
    - Start with small inputs/tasks, then scale once behavior is verified.
