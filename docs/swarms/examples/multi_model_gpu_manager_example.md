# Multi Model Gpu Manager Tutorial

    End-to-end usage for `multi_model_gpu_manager`.

    ## Prerequisites

    - Python 3.10+
    - `pip install -U swarms`
    - Provider credentials configured when using hosted LLMs

    ## Example

    ```python
    import torch
from swarms.structs.multi_model_gpu_manager import ModelGrid, GPUAllocationStrategy

# Minimal local model (example)
model = torch.nn.Linear(128, 16)

grid = ModelGrid(allocation_strategy=GPUAllocationStrategy.MEMORY_OPTIMIZED)
grid.add_model(model_name="linear-demo", model=model)

grid.allocate_all_models()
grid.load_all_models()

result = grid.run(model_name="linear-demo", task="forward", input_data=torch.randn(1, 128))
print(result)

grid.unload_all_models()
    ```

    ## What this demonstrates

    - Basic construction/import pattern for `multi_model_gpu_manager`
    - Minimal execution path you can adapt in production
    - Safe starting defaults for iteration

    ## Related

    - Struct reference: [`swarms/structs/multi_model_gpu_manager.md`](../structs/multi_model_gpu_manager.md)
