# Agent Output Types Examples with Vision Capabilities

This example demonstrates how to use different output types when working with Swarms agents, including vision-enabled agents that can analyze images. Each output type formats the agent's response in a specific way, making it easier to integrate with different parts of your application.

## Prerequisites

- Python 3.7+
- OpenAI API key
- Anthropic API key (optional, for Claude models)
- Swarms library

## Installation

```bash
pip3 install -U swarms
```

## Environment Variables

```plaintext
WORKSPACE_DIR="agent_workspace"
OPENAI_API_KEY=""  # Required for GPT-4V vision capabilities
ANTHROPIC_API_KEY=""  # Optional, for Claude models
```

## Examples

### Vision-Enabled Quality Control Agent

```python
from swarms.structs import Agent
from swarms.prompts.logistics import (
    Quality_Control_Agent_Prompt,
)

# Image for analysis
factory_image = "image.jpg"


# Quality control agent
quality_control_agent = Agent(
    agent_name="Quality Control Agent",
    agent_description="A quality control agent that analyzes images and provides a detailed report on the quality of the product in the image.",
    model_name="gpt-4.1-mini",
    system_prompt=Quality_Control_Agent_Prompt,
    multi_modal=True,
    max_loops=2,
    output_type="str-all-except-first",
)


response = quality_control_agent.run(
    task="what is in the image?",
    img=factory_image,
)

print(response)

```

### Supported Image Formats

The vision-enabled agents support various image formats including:

| Format | Description |
|--------|-------------|
| JPEG/JPG | Standard image format with lossy compression |
| PNG | Lossless format supporting transparency |
| GIF | Animated format (only first frame used) |
| WebP | Modern format with both lossy and lossless compression |

### Best Practices for Vision Tasks

| Best Practice | Description |
|--------------|-------------|
| Image Quality | Ensure images are clear and well-lit for optimal analysis |
| Image Size | Keep images under 20MB and in supported formats |
| Task Specificity | Provide clear, specific instructions for image analysis |
| Model Selection | Use vision-capable models (e.g., GPT-4V) for image tasks |