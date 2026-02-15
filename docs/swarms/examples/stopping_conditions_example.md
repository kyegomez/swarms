# Stopping Conditions Tutorial

    End-to-end usage for `stopping_conditions`.

    ## Prerequisites

    - Python 3.10+
    - `pip install -U swarms`
    - Provider credentials configured when using hosted LLMs

    ## Example

    ```python
    from swarms.structs.stopping_conditions import check_stopping_conditions

samples = [
    "analysis complete",
    "execution finished",
    "error: timeout",
    "continue processing",
]

for text in samples:
    print(text, "->", check_stopping_conditions(text))
    ```

    ## What this demonstrates

    - Basic construction/import pattern for `stopping_conditions`
    - Minimal execution path you can adapt in production
    - Safe starting defaults for iteration

    ## Related

    - Struct reference: [`swarms/structs/stopping_conditions.md`](../structs/stopping_conditions.md)
