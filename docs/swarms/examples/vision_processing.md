# Vision Processing Examples

This example demonstrates how to use vision-enabled agents in Swarms to analyze images and process visual information. You'll learn how to work with both OpenAI and Anthropic vision models for various use cases.

## Prerequisites

- Python 3.7+

- OpenAI API key (for GPT-4V)

- Anthropic API key (for Claude 3)

- Swarms library

## Installation

```bash
pip3 install -U swarms
```

## Environment Variables

```plaintext
WORKSPACE_DIR="agent_workspace"
OPENAI_API_KEY=""  # Required for GPT-4V
ANTHROPIC_API_KEY=""  # Required for Claude 3
```

## Working with Images

### Supported Image Formats

Vision-enabled agents support various image formats:

| Format | Description |
|--------|-------------|
| JPEG/JPG | Standard image format with lossy compression |
| PNG | Lossless format supporting transparency |
| GIF | Animated format (only first frame used) |
| WebP | Modern format with both lossy and lossless compression |

### Image Guidelines

- Maximum file size: 20MB
- Recommended resolution: At least 512x512 pixels
- Image should be clear and well-lit
- Avoid heavily compressed or blurry images

## Examples

### 1. Quality Control with GPT-4V

```python
from swarms.structs import Agent
from swarms.prompts.logistics import Quality_Control_Agent_Prompt

# Load your image
factory_image = "path/to/your/image.jpg"  # Local file path
# Or use a URL
# factory_image = "https://example.com/image.jpg"

# Initialize quality control agent with GPT-4V
quality_control_agent = Agent(
    agent_name="Quality Control Agent",
    agent_description="A quality control agent that analyzes images and provides detailed quality reports.",
    model_name="gpt-4.1-mini",
    system_prompt=Quality_Control_Agent_Prompt,
    multi_modal=True,
    max_loops=1
)

# Run the analysis
response = quality_control_agent.run(
    task="Analyze this image and provide a detailed quality control report",
    img=factory_image
)

print(response)
```

### 2. Visual Analysis with Claude 3

```python
from swarms.structs import Agent
from swarms.prompts.logistics import Visual_Analysis_Prompt

# Load your image
product_image = "path/to/your/product.jpg"

# Initialize visual analysis agent with Claude 3
visual_analyst = Agent(
    agent_name="Visual Analyst",
    agent_description="An agent that performs detailed visual analysis of products and scenes.",
    model_name="anthropic/claude-3-opus-20240229",
    system_prompt=Visual_Analysis_Prompt,
    multi_modal=True,
    max_loops=1
)

# Run the analysis
response = visual_analyst.run(
    task="Provide a comprehensive analysis of this product image",
    img=product_image
)

print(response)
```

### 3. Image Batch Processing

```python
from swarms.structs import Agent
import os

def process_image_batch(image_folder, agent):
    """Process multiple images in a folder"""
    results = []
    for image_file in os.listdir(image_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_path = os.path.join(image_folder, image_file)
            response = agent.run(
                task="Analyze this image",
                img=image_path
            )
            results.append((image_file, response))
    return results

# Example usage
image_folder = "path/to/image/folder"
batch_results = process_image_batch(image_folder, visual_analyst)
```

## Best Practices

| Category | Best Practice | Description |
|----------|---------------|-------------|
| Image Preparation | Format Support | Ensure images are in supported formats (JPEG, PNG, GIF, WebP) |
| | Size & Quality | Optimize image size and quality for better processing |
| | Image Quality | Use clear, well-lit images for accurate analysis |
| Model Selection | GPT-4V Usage | Use for general vision tasks and detailed analysis |
| | Claude 3 Usage | Use for complex reasoning and longer outputs |
| | Batch Processing | Consider batch processing for multiple images |
| Error Handling | Path Validation | Always validate image paths before processing |
| | API Error Handling | Implement proper error handling for API calls |
| | Rate Monitoring | Monitor API rate limits and token usage |
| Performance Optimization | Result Caching | Cache results when processing the same images |
| | Batch Processing | Use batch processing for multiple images |
| | Parallel Processing | Implement parallel processing for large datasets |


