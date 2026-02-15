# Image Batch Processor

    `image_batch_processor` reference documentation.

    **Module Path**: `swarms.structs.image_batch_processor`

    ## Overview

    Parallel image-processing orchestrator for running one or many agents across image files and tasks.

    ## Public API

    - **`ImageProcessingError`**: No public methods documented in this module.
- **`InvalidAgentError`**: No public methods documented in this module.
- **`ImageAgentBatchProcessor`**: `run()`

    ## Quickstart

    ```python
    from swarms.structs.image_batch_processor import ImageProcessingError, InvalidAgentError, ImageAgentBatchProcessor
    ```

    ## Tutorial

    A runnable tutorial is available at [`swarms/examples/image_batch_processor_example.md`](../examples/image_batch_processor_example.md).

    ## Notes

    - Keep task payloads small for first runs.
    - Prefer deterministic prompts when comparing outputs across agents.
    - Validate provider credentials (for LLM-backed examples) before production use.
