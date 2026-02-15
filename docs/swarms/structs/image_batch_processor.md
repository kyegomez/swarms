# Image Batch Processor

    Reference documentation for `swarms.structs.image_batch_processor`.

    ## Overview

    This module provides production utilities for `image batch processor` in Swarms.

    ## Module Path

    ```python
    from swarms.structs.image_batch_processor import ...
    ```

    ## Public API

    - `ImageAgentBatchProcessor.run(image_paths, tasks)` and callable shortcut via `__call__`

    ## Quick Start

    ```python
    from swarms import Agent
from swarms.structs.image_batch_processor import ImageAgentBatchProcessor

vision_agent = Agent(agent_name="Vision", model_name="gpt-4.1")
processor = ImageAgentBatchProcessor(agents=[vision_agent], max_workers=2)
print(processor.run(image_paths=["./sample.jpg"], tasks=["Describe the image"]))
    ```

    ## Tutorial

    See the runnable tutorial: [`swarms/examples/image_batch_processor_example.md`](../examples/image_batch_processor_example.md)

    ## Operational Notes

    - Validate credentials and model access before running LLM-backed examples.
    - Start with small inputs/tasks, then scale once behavior is verified.
- All image paths must exist and use supported suffixes (`.jpg`, `.jpeg`, `.png`) unless overridden.
