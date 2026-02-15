# Safe Loading Tutorial

    End-to-end tutorial for `swarms.structs.safe_loading`.

    ## Prerequisites

    - Python 3.10+
    - `pip install -U swarms`

    ## Example

    ```python
    from swarms.structs.safe_loading import SafeStateManager

class Config:
    def __init__(self):
        self.name = "demo"
        self.version = 1

cfg = Config()
SafeStateManager.save_state(cfg, "./state/config.json")
cfg.version = 2
SafeStateManager.load_state(cfg, "./state/config.json")
print(cfg.version)
    ```

    ## What this demonstrates

    - Correct import and initialization flow for `safe_loading`
    - Minimal execution path suitable for first integration tests
    - A baseline pattern to adapt for production use

    ## Related

    - Struct reference: [`swarms/structs/safe_loading.md`](../structs/safe_loading.md)
