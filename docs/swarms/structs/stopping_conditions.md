# Stopping Conditions

    `stopping_conditions` reference documentation.

    **Module Path**: `swarms.structs.stopping_conditions`

    ## Overview

    Keyword-based stopping-condition predicates and a unified evaluator for workflow termination checks.

    ## Public API

    - **`check_done()`**
- **`check_finished()`**
- **`check_complete()`**
- **`check_success()`**
- **`check_failure()`**
- **`check_error()`**
- **`check_stopped()`**
- **`check_cancelled()`**
- **`check_exit()`**
- **`check_end()`**
- **`check_stopping_conditions()`**

    ## Quickstart

    ```python
    from swarms.structs.stopping_conditions import check_done, check_finished, check_complete
    ```

    ## Tutorial

    A runnable tutorial is available at [`swarms/examples/stopping_conditions_example.md`](../examples/stopping_conditions_example.md).

    ## Notes

    - Keep task payloads small for first runs.
    - Prefer deterministic prompts when comparing outputs across agents.
    - Validate provider credentials (for LLM-backed examples) before production use.
