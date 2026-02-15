# Image Batch Processor Tutorial

    End-to-end tutorial for `swarms.structs.image_batch_processor`.

    ## Prerequisites

    - Python 3.10+
    - `pip install -U swarms`

    ## Example

    ```python
    from swarms import Agent
from swarms.structs.image_batch_processor import ImageAgentBatchProcessor

vision_agent = Agent(agent_name="Vision", model_name="gpt-4.1")
processor = ImageAgentBatchProcessor(agents=[vision_agent], max_workers=2)
print(processor.run(image_paths=["./sample.jpg"], tasks=["Describe the image"]))
    ```

    ## What this demonstrates

    - Correct import and initialization flow for `image_batch_processor`
    - Minimal execution path suitable for first integration tests
    - A baseline pattern to adapt for production use

    ## Related

    - Struct reference: [`swarms/structs/image_batch_processor.md`](../structs/image_batch_processor.md)
