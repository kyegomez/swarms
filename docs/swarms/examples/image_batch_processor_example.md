# Image Batch Processor Tutorial

    End-to-end usage for `image_batch_processor`.

    ## Prerequisites

    - Python 3.10+
    - `pip install -U swarms`
    - Provider credentials configured when using hosted LLMs

    ## Example

    ```python
    from swarms import Agent
from swarms.structs.image_batch_processor import ImageAgentBatchProcessor

vision_agent = Agent(
    agent_name="Vision-Analyst",
    system_prompt="Describe and analyze images.",
    model_name="gpt-4.1",
    max_loops=1,
)

processor = ImageAgentBatchProcessor(agents=vision_agent)
results = processor.run(
    image_paths=["./examples/sample1.png", "./examples/sample2.png"],
    tasks=["Describe key objects", "Identify risks"],
)
print(results)
    ```

    ## What this demonstrates

    - Basic construction/import pattern for `image_batch_processor`
    - Minimal execution path you can adapt in production
    - Safe starting defaults for iteration

    ## Related

    - Struct reference: [`swarms/structs/image_batch_processor.md`](../structs/image_batch_processor.md)
