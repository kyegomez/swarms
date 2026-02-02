# Agent with Base64 Image (Data URI)

This example shows how to run a vision agent using a **base64 data URI** for an image. Use this when you have a local image file and want to pass it to the agent without hosting a URL.

## Prerequisites

- Python 3.7+
- OpenAI API key (for GPT-4V or compatible vision model)
- Swarms library

## Installation

```bash
pip install -U swarms
```

## Environment

Set your API key:

```plaintext
OPENAI_API_KEY=""
```

## Code

1. Create an agent with a vision-capable model (e.g. `gpt-4.1`).
2. Use `get_image_data_uri()` to load a local image as a base64 data URI.
3. Pass the data URI to `agent.run()` via the `img` parameter.

```python
from swarms import Agent
from swarms.utils.image_file_b64 import get_image_data_uri

# Initialize agent
agent = Agent(
    model_name="gpt-4.1",
    max_loops=1,
    verbose=True,
)

# Load image as data URI (base64 with data URI prefix)
data_uri = get_image_data_uri("image.jpg")

task = "Where does this image come from?"
result = agent.run(task=task, img=data_uri)

print(result)
```

Replace `"image.jpg"` with the path to your image file. The agent will receive the image as a data URI and answer the given task.

## Summary

| Step | Action |
|------|--------|
| 1 | Import `Agent` and `get_image_data_uri` |
| 2 | Create an `Agent` with a vision model |
| 3 | Call `get_image_data_uri("path/to/image.jpg")` |
| 4 | Run `agent.run(task="...", img=data_uri)` |

For multiple images or URL-based images, see [Agents with Vision](vision_processing.md) and [Agent with Multiple Images](multiple_images.md).
