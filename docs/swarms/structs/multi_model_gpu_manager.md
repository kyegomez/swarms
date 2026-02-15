# Multi Model GPU Manager

    Reference documentation for `swarms.structs.multi_model_gpu_manager`.

    ## Overview

    This module provides production utilities for `multi model gpu manager` in Swarms.

    ## Module Path

    ```python
    from swarms.structs.multi_model_gpu_manager import ...
    ```

    ## Public API

    - `ModelGrid`: `add_model`, `allocate_all_models`, `load_all_models`, `run`, `get_gpu_status`

    ## Quick Start

    ```python
    from swarms.structs.multi_model_gpu_manager import ModelGrid, GPUAllocationStrategy

grid = ModelGrid(allocation_strategy=GPUAllocationStrategy.MEMORY_OPTIMIZED, use_multiprocessing=False)
print(grid.get_gpu_status())
    ```

    ## Tutorial

    See the runnable tutorial: [`swarms/examples/multi_model_gpu_manager_example.md`](../examples/multi_model_gpu_manager_example.md)

    ## Operational Notes

    - Validate credentials and model access before running LLM-backed examples.
    - Start with small inputs/tasks, then scale once behavior is verified.
- This module expects CUDA/PyTorch runtime for full GPU orchestration. Start with status calls before loading models.
