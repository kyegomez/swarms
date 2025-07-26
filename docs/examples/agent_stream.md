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

