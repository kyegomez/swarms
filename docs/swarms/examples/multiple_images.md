# Processing Multiple Images

This tutorial shows how to process multiple images with a single agent using Swarms' multi-modal capabilities. You'll learn to configure an agent for batch image analysis, enabling efficient processing for quality control, object detection, or image comparison tasks.


## Installation

Install the swarms package using pip:

```bash
pip install -U swarms
```

## Basic Setup

1. First, set up your environment variables:

```python
WORKSPACE_DIR="agent_workspace"
ANTHROPIC_API_KEY=""
```


## Code

- Create a list of images by their file paths

- Pass it into the `Agent.run(imgs=[str])` parameter

- Activate `summarize_multiple_images=True` if you want the agent to output a summary of the image analyses


```python
from swarms import Agent
from swarms.prompts.logistics import (
    Quality_Control_Agent_Prompt,
)


# Image for analysis
factory_image = "image.jpg"

# Quality control agent
quality_control_agent = Agent(
    agent_name="Quality Control Agent",
    agent_description="A quality control agent that analyzes images and provides a detailed report on the quality of the product in the image.",
    model_name="claude-3-5-sonnet-20240620",
    system_prompt=Quality_Control_Agent_Prompt,
    multi_modal=True,
    max_loops=1,
    output_type="str-all-except-first",
    summarize_multiple_images=True,
)


response = quality_control_agent.run(
    task="what is in the image?",
    imgs=[factory_image, factory_image],
)

print(response)
```

## Support and Community

If you're facing issues or want to learn more, check out the following resources to join our Discord, stay updated on Twitter, and watch tutorials on YouTube!

| Platform | Link | Description |
|----------|------|-------------|
| 📚 Documentation | [docs.swarms.world](https://docs.swarms.world) | Official documentation and guides |
| 📝 Blog | [Medium](https://medium.com/@kyeg) | Latest updates and technical articles |
| 💬 Discord | [Join Discord](https://discord.gg/EamjgSaEQf) | Live chat and community support |
| 🐦 Twitter | [@kyegomez](https://twitter.com/kyegomez) | Latest news and announcements |
| 👥 LinkedIn | [The Swarm Corporation](https://www.linkedin.com/company/the-swarm-corporation) | Professional network and updates |
| 📺 YouTube | [Swarms Channel](https://www.youtube.com/channel/UC9yXyitkbU_WSy7bd_41SqQ) | Tutorials and demos |
| 🎫 Events | [Sign up here](https://lu.ma/swarms_calendar) | Join our community events |

