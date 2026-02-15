# Multi Model GPU Manager Tutorial

    End-to-end tutorial for `swarms.structs.multi_model_gpu_manager`.

    ## Prerequisites

    - Python 3.10+
    - `pip install -U swarms`

    ## Example

    ```python
    from swarms.structs.multi_model_gpu_manager import ModelGrid, GPUAllocationStrategy

grid = ModelGrid(allocation_strategy=GPUAllocationStrategy.MEMORY_OPTIMIZED, use_multiprocessing=False)
print(grid.get_gpu_status())
    ```

    ## What this demonstrates

    - Correct import and initialization flow for `multi_model_gpu_manager`
    - Minimal execution path suitable for first integration tests
    - A baseline pattern to adapt for production use

    ## Related

    - Struct reference: [`swarms/structs/multi_model_gpu_manager.md`](../structs/multi_model_gpu_manager.md)
