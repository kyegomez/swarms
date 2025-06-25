# Agents with Vision and Tool Usage

This tutorial demonstrates how to create intelligent agents that can analyze images and use custom tools to perform specific actions based on their visual observations. You'll learn to build a quality control agent that can process images, identify potential security concerns, and automatically trigger appropriate responses using function calling capabilities.

## What You'll Learn

- How to configure an agent with multi-modal capabilities for image analysis
- How to integrate custom tools and functions with vision-enabled agents
- How to implement automated security analysis based on visual observations
- How to use function calling to trigger specific actions from image analysis results
- Best practices for building production-ready vision agents with tool integration

## Use Cases

This approach is perfect for:

- **Quality Control Systems**: Automated inspection of manufacturing processes

- **Security Monitoring**: Real-time threat detection and response

- **Object Detection**: Identifying and categorizing items in images

- **Compliance Checking**: Ensuring standards are met in various environments

- **Automated Reporting**: Generating detailed analysis reports from visual data

## Installation

Install the swarms package using pip:

```bash
pip install -U swarms
```

## Basic Setup

1. First, set up your environment variables:

```python
WORKSPACE_DIR="agent_workspace"
OPENAI_API_KEY=""
```


## Code

- Create tools for your agent as a function with types and documentation

- Pass tools to your agent `Agent(tools=[list_of_callables])`

- Add your image path to the run method like: `Agent().run(task=task, img=img)`

- 

```python
from swarms.structs import Agent
from swarms.prompts.logistics import (
    Quality_Control_Agent_Prompt,
)


# Image for analysis
factory_image = "image.jpg"


def security_analysis(danger_level: str) -> str:
    """
    Analyzes the security danger level and returns an appropriate response.

    Args:
        danger_level (str, optional): The level of danger to analyze.
            Can be "low", "medium", "high", or None. Defaults to None.

    Returns:
        str: A string describing the danger level assessment.
            - "No danger level provided" if danger_level is None
            - "No danger" if danger_level is "low"
            - "Medium danger" if danger_level is "medium"
            - "High danger" if danger_level is "high"
            - "Unknown danger level" for any other value
    """
    if danger_level is None:
        return "No danger level provided"

    if danger_level == "low":
        return "No danger"

    if danger_level == "medium":
        return "Medium danger"

    if danger_level == "high":
        return "High danger"

    return "Unknown danger level"


custom_system_prompt = f"""
{Quality_Control_Agent_Prompt}

You have access to tools that can help you with your analysis. When you need to perform a security analysis, you MUST use the security_analysis function with an appropriate danger level (low, medium, or high) based on your observations.

Always use the available tools when they are relevant to the task. If you determine there is any level of danger or security concern, call the security_analysis function with the appropriate danger level.
"""

# Quality control agent
quality_control_agent = Agent(
    agent_name="Quality Control Agent",
    agent_description="A quality control agent that analyzes images and provides a detailed report on the quality of the product in the image.",
    # model_name="anthropic/claude-3-opus-20240229",
    model_name="gpt-4o-mini",
    system_prompt=custom_system_prompt,
    multi_modal=True,
    max_loops=1,
    output_type="str-all-except-first",
    # tools_list_dictionary=[schema],
    tools=[security_analysis],
)


response = quality_control_agent.run(
    task="Analyze the image and then perform a security analysis. Based on what you see in the image, determine if there is a low, medium, or high danger level and call the security_analysis function with that danger level",
    img=factory_image,
)
```


## Support and Community

If you're facing issues or want to learn more, check out the following resources to join our Discord, stay updated on Twitter, and watch tutorials on YouTube!

| Platform | Link | Description |
|----------|------|-------------|
| 📚 Documentation | [docs.swarms.world](https://docs.swarms.world) | Official documentation and guides |
| 📝 Blog | [Medium](https://medium.com/@kyeg) | Latest updates and technical articles |
| 💬 Discord | [Join Discord](https://discord.gg/jM3Z6M9uMq) | Live chat and community support |
| 🐦 Twitter | [@kyegomez](https://twitter.com/kyegomez) | Latest news and announcements |
| 👥 LinkedIn | [The Swarm Corporation](https://www.linkedin.com/company/the-swarm-corporation) | Professional network and updates |
| 📺 YouTube | [Swarms Channel](https://www.youtube.com/channel/UC9yXyitkbU_WSy7bd_41SqQ) | Tutorials and demos |
| 🎫 Events | [Sign up here](https://lu.ma/5p2jnc2v) | Join our community events |

