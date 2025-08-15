# Agent with Streaming

The Swarms framework provides powerful real-time streaming capabilities for agents, allowing you to see responses being generated token by token as they're produced by the language model. This creates a more engaging and interactive experience, especially useful for long-form content generation, debugging, or when you want to provide immediate feedback to users.

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

## Step by Step

- Install and put your keys in `.env`

- Turn on streaming in `Agent()` with `streaming_on=True`

- Optional: If you want to pretty print it, you can do `print_on=True`; if not, it will print normally

## Code

```python
from swarms import Agent

# Enable real-time streaming
agent = Agent(
    agent_name="StoryAgent",
    model_name="gpt-4o-mini",
    streaming_on=True,  # 🔥 This enables real streaming!
    max_loops=1,
    print_on=True,  # By default, it's False for raw streaming!
)

# This will now stream in real-time with a beautiful UI!
response = agent.run("Tell me a detailed story about humanity colonizing the stars")
print(response)
```

## Streaming with Tools Execution

Swarms also supports real-time streaming while executing tools, providing immediate feedback on both the thinking process and tool execution results:

```python
from swarms import Agent

def get_weather(location: str, units: str = "celsius") -> str:
    """
    Get the current weather for a location.
    
    Args:
        location (str): The city/location to get weather for
        units (str): Temperature units (celsius or fahrenheit)

    Returns:
        str: Weather information
    """
    weather_data = {
        "New York": {"temperature": "22°C", "condition": "sunny", "humidity": "65%"},
        "London": {"temperature": "15°C", "condition": "cloudy", "humidity": "80%"},
        "Tokyo": {"temperature": "28°C", "condition": "rainy", "humidity": "90%"},
    }
    
    location_key = location.title()
    if location_key in weather_data:
        data = weather_data[location_key]
        temp = data["temperature"] 
        if units == "fahrenheit" and "°C" in temp:
            celsius = int(temp.replace("°C", ""))
            fahrenheit = (celsius * 9/5) + 32
            temp = f"{fahrenheit}°F"
        
        return f"Weather in {location}: {temp}, {data['condition']}, humidity: {data['humidity']}"
    else:
        return f"Weather data not available for {location}"

# Create agent with streaming and tool support
agent = Agent(
    model_name="gpt-4o",
    max_loops=1,
    verbose=True,
    streaming_on=True,  # Enable streaming
    print_on=True,      # Enable pretty printing
    tools=[get_weather], # Add tools
)

# This will stream both the reasoning and tool execution results
agent.run("What is the weather in Tokyo? ")
```

### Key Features of Streaming with Tools:

- **Real-time tool execution**: See tool calls happen as they're invoked
- **Streaming responses**: Get immediate feedback on the agent's reasoning
- **Tool result integration**: Watch how tools results are incorporated into the final response
- **Interactive debugging**: Monitor the complete workflow from thought to action

### Best Practices:

1. **Set appropriate max_loops**: Use `max_loops=1` for simple tasks or higher values for complex multi-step operations
2. **Enable verbose mode**: Use `verbose=True` to see detailed tool execution logs
3. **Use print_on for UI**: Enable `print_on=True` for better visual streaming experience
4. **Monitor performance**: Streaming with tools may be slower due to real-time processing

## Connect With Us

If you'd like technical support, join our Discord below and stay updated on our Twitter for new updates!

| Platform | Link | Description |
|----------|------|-------------|
| 📚 Documentation | [docs.swarms.world](https://docs.swarms.world) | Official documentation and guides |
| 📝 Blog | [Medium](https://medium.com/@kyeg) | Latest updates and technical articles |
| 💬 Discord | [Join Discord](https://discord.gg/jM3Z6M9uMq) | Live chat and community support |
| 🐦 Twitter | [@kyegomez](https://twitter.com/kyegomez) | Latest news and announcements |
| 👥 LinkedIn | [The Swarm Corporation](https://www.linkedin.com/company/the-swarm-corporation) | Professional network and updates |
| 📺 YouTube | [Swarms Channel](https://www.youtube.com/channel/UC9yXyitkbU_WSy7bd_41SqQ) | Tutorials and demos |
| 🎫 Events | [Sign up here](https://lu.ma/5p2jnc2v) | Join our community events |

