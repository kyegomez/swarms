# Safe Loading Tutorial

    End-to-end usage for `safe_loading`.

    ## Prerequisites

    - Python 3.10+
    - `pip install -U swarms`
    - Provider credentials configured when using hosted LLMs

    ## Example

    ```python
    from swarms.structs.safe_loading import SafeStateManager

class WorkflowState:
    def __init__(self):
        self.name = "pipeline-a"
        self.retries = 2
        self.tags = ["etl", "daily"]

state = WorkflowState()
SafeStateManager.save_state(state, "./state/workflow_state.json")

state.retries = 0
SafeStateManager.load_state(state, "./state/workflow_state.json")
print(state.retries)  # 2
    ```

    ## What this demonstrates

    - Basic construction/import pattern for `safe_loading`
    - Minimal execution path you can adapt in production
    - Safe starting defaults for iteration

    ## Related

    - Struct reference: [`swarms/structs/safe_loading.md`](../structs/safe_loading.md)
