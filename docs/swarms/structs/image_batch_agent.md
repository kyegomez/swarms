# ImageAgentBatchProcessor Documentation

## Overview

The `ImageAgentBatchProcessor` is a high-performance parallel image processing system designed for running AI agents on multiple images concurrently. It provides robust error handling, logging, and flexible configuration options.

## Installation

```bash
pip install swarms
```

## Class Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| agents | Union[Agent, List[Agent], Callable, List[Callable]] | Required | Single agent or list of agents to process images |
| max_workers | int | None | Maximum number of parallel workers (defaults to 95% of CPU cores) |
| supported_formats | List[str] | ['.jpg', '.jpeg', '.png'] | List of supported image file extensions |

## Methods

### run()

**Description**: Main method for processing multiple images in parallel with configured agents. Can handle single images, multiple images, or entire directories.

**Arguments**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| image_paths | Union[str, List[str], Path] | Yes | Single image path, list of paths, or directory path |
| tasks | Union[str, List[str]] | Yes | Single task or list of tasks to perform on each image |

**Returns**: List[Dict[str, Any]] - List of processing results for each image

**Example**:

```python
from swarms import Agent
from swarms.structs import ImageAgentBatchProcessor
from pathlib import Path

# Initialize agent and processor
agent = Agent(api_key="your-api-key", model="gpt-4-vision")
processor = ImageAgentBatchProcessor(agents=agent)

# Example 1: Process single image
results = processor.run(
    image_paths="path/to/image.jpg",
    tasks="Describe this image"
)

# Example 2: Process multiple images
results = processor.run(
    image_paths=["image1.jpg", "image2.jpg"],
    tasks=["Describe objects", "Identify colors"]
)

# Example 3: Process directory
results = processor.run(
    image_paths=Path("./images"),
    tasks="Analyze image content"
)
```

### _validate_image_path()

**Description**: Internal method that validates if an image path exists and has a supported format.

**Arguments**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| image_path | Union[str, Path] | Yes | Path to the image file to validate |

**Returns**: Path - Validated Path object

**Example**:
```python

from swarms.structs import ImageAgentBatchProcessor, ImageProcessingError
from pathlib import Path

processor = ImageAgentBatchProcessor(agents=agent)

try:
    validated_path = processor._validate_image_path("image.jpg")
    print(f"Valid image path: {validated_path}")
except ImageProcessingError as e:
    print(f"Invalid image path: {e}")
```

### _process_single_image()

**Description**: Internal method that processes a single image with one agent and one or more tasks.

**Arguments**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| image_path | Path | Yes | Path to the image to process |
| tasks | Union[str, List[str]] | Yes | Tasks to perform on the image |
| agent | Agent | Yes | Agent to use for processing |

**Returns**: Dict[str, Any] - Processing results for the image

**Example**:

```python
from swarms import Agent
from swarms.structs import ImageAgentBatchProcessor
from pathlib import Path

agent = Agent(api_key="your-api-key", model="gpt-4-vision")
processor = ImageAgentBatchProcessor(agents=agent)

try:
    result = processor._process_single_image(
        image_path=Path("image.jpg"),
        tasks=["Describe image", "Identify objects"],
        agent=agent
    )
    print(f"Processing results: {result}")
except Exception as e:
    print(f"Processing failed: {e}")
```

### __call__()

**Description**: Makes the ImageAgentBatchProcessor callable like a function. Redirects to the run() method.

**Arguments**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| *args | Any | No | Variable length argument list passed to run() |
| **kwargs | Any | No | Keyword arguments passed to run() |

**Returns**: List[Dict[str, Any]] - Same as run() method

**Example**:

```python
from swarms import Agent
from swarms.structs import ImageAgentBatchProcessor

# Initialize
agent = Agent(api_key="your-api-key", model="gpt-4-vision")
processor = ImageAgentBatchProcessor(agents=agent)

# Using __call__
results = processor(
    image_paths=["image1.jpg", "image2.jpg"],
    tasks="Describe the image"
)

# This is equivalent to:
results = processor.run(
    image_paths=["image1.jpg", "image2.jpg"],
    tasks="Describe the image"
)
```

## Return Format

The processor returns a list of dictionaries with the following structure:

```python
{
    "image_path": str,          # Path to the processed image
    "results": {               # Results for each task
        "task_name": result,   # Task-specific results
    },
    "processing_time": float   # Processing time in seconds
}
```

## Complete Usage Examples

### 1. Basic Usage with Single Agent

```python
from swarms import Agent
from swarms.structs import ImageAgentBatchProcessor

# Initialize an agent
agent = Agent(
    api_key="your-api-key",
    model="gpt-4-vision"
)

# Create processor
processor = ImageAgentBatchProcessor(agents=agent)

# Process single image
results = processor.run(
    image_paths="path/to/image.jpg",
    tasks="Describe this image in detail"
)
```

### 2. Processing Multiple Images with Multiple Tasks

```python
# Initialize with multiple agents
agent1 = Agent(api_key="key1", model="gpt-4-vision")
agent2 = Agent(api_key="key2", model="claude-3")

processor = ImageAgentBatchProcessor(
    agents=[agent1, agent2],
    supported_formats=['.jpg', '.png', '.webp']
)

# Define multiple tasks
tasks = [
    "Describe the main objects in the image",
    "What is the dominant color?",
    "Identify any text in the image"
]

# Process a directory of images
results = processor.run(
    image_paths="path/to/image/directory",
    tasks=tasks
)

# Process results
for result in results:
    print(f"Image: {result['image_path']}")
    for task, output in result['results'].items():
        print(f"Task: {task}")
        print(f"Result: {output}")
    print(f"Processing time: {result['processing_time']:.2f} seconds")
```

### 3. Custom Error Handling

```python
from swarms.structs import ImageAgentBatchProcessor, ImageProcessingError

try:
    processor = ImageAgentBatchProcessor(agents=agent)
    results = processor.run(
        image_paths=["image1.jpg", "image2.png", "invalid.txt"],
        tasks="Analyze the image"
    )
except ImageProcessingError as e:
    print(f"Image processing failed: {e}")
except InvalidAgentError as e:
    print(f"Agent configuration error: {e}")
```

## Best Practices

| Best Practice | Description |
|--------------|-------------|
| Resource Management | • The processor automatically uses 95% of available CPU cores<br>• For memory-intensive operations, consider reducing `max_workers` |
| Error Handling | • Always wrap processor calls in try-except blocks<br>• Check the results for any error keys |
| Task Design | • Keep tasks focused and specific<br>• Group related tasks together for efficiency |
| Performance Optimization | • Process images in batches for better throughput<br>• Use multiple agents for different types of analysis |

## Limitations

| Limitation | Description |
|------------|-------------|
| File Format Support | Only supports image file formats specified in `supported_formats` |
| Agent Requirements | Requires valid agent configurations |
| Resource Scaling | Memory usage scales with number of concurrent processes |


This documentation provides a comprehensive guide to using the `ImageAgentBatchProcessor`. The class is designed to be both powerful and flexible, allowing for various use cases from simple image analysis to complex multi-agent processing pipelines.
