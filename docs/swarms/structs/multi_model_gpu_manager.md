# Multi Model Gpu Manager

    `multi_model_gpu_manager` reference documentation.

    **Module Path**: `swarms.structs.multi_model_gpu_manager`

    ## Overview

    Production utility for allocating and running many ML models across available GPUs with memory-aware placement.

    ## Public API

    - **`ModelType`**: No public methods documented in this module.
- **`GPUAllocationStrategy`**: No public methods documented in this module.
- **`ModelMetadata`**: No public methods documented in this module.
- **`GPUMetadata`**: No public methods documented in this module.
- **`ModelMemoryCalculator`**: `get_pytorch_model_size()`, `get_huggingface_model_size()`
- **`GPUManager`**: `update_gpu_memory_info()`
- **`ModelGrid`**: `add_model()`, `remove_model()`, `allocate_all_models()`, `load_model()`, `unload_model()`, `load_all_models()`, `unload_all_models()`, `run()`
- **`ModelWithCustomRunMethod`**: `run()`
- **`PyTorchModelWrapper`**: `run()`
- **`HuggingFaceModelWrapper`**: `run()`

    ## Quickstart

    ```python
    from swarms.structs.multi_model_gpu_manager import ModelType, GPUAllocationStrategy, ModelMetadata
    ```

    ## Tutorial

    A runnable tutorial is available at [`swarms/examples/multi_model_gpu_manager_example.md`](../examples/multi_model_gpu_manager_example.md).

    ## Notes

    - Keep task payloads small for first runs.
    - Prefer deterministic prompts when comparing outputs across agents.
    - Validate provider credentials (for LLM-backed examples) before production use.
