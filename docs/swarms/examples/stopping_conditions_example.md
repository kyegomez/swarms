# Stopping Conditions Tutorial

    End-to-end tutorial for `swarms.structs.stopping_conditions`.

    ## Prerequisites

    - Python 3.10+
    - `pip install -U swarms`

    ## Example

    ```python
    from swarms.structs.stopping_conditions import check_stopping_conditions

print(check_stopping_conditions("task finished successfully"))
print(check_stopping_conditions("continue running"))
    ```

    ## What this demonstrates

    - Correct import and initialization flow for `stopping_conditions`
    - Minimal execution path suitable for first integration tests
    - A baseline pattern to adapt for production use

    ## Related

    - Struct reference: [`swarms/structs/stopping_conditions.md`](../structs/stopping_conditions.md)
