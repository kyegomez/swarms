from swarms import Agent
from swarms.structs.image_batch_processor import (
    ImageAgentBatchProcessor,
)
from pathlib import Path

# Initialize agent and processor

# Quality control agent
agent = Agent(
    model_name="gpt-4.1-mini",
    max_loops=1,
)

# Create processor
processor = ImageAgentBatchProcessor(agents=agent)

# Example 1: Process single image
results = processor.run(
    image_paths="path/to/image.jpg", tasks="Describe this image"
)

# Example 2: Process multiple images
results = processor.run(
    image_paths=["image1.jpg", "image2.jpg"],
    tasks=["Describe objects", "Identify colors"],
)

# Example 3: Process directory
results = processor.run(
    image_paths=Path("./images"), tasks="Analyze image content"
)
