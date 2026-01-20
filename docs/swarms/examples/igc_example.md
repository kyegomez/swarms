## Interactive Groupchat Examples

The Interactive GroupChat is a powerful multi-agent architecture that enables dynamic collaboration between multiple AI agents. This architecture allows agents to communicate with each other, respond to mentions using `@agent_name` syntax, and work together to solve complex tasks through structured conversation flows.

### Architecture Description

The Interactive GroupChat implements a **collaborative swarm architecture** where multiple specialized agents work together in a coordinated manner. Key features include:

- **Mention-based Communication**: Agents can be directed to specific tasks using `@agent_name` syntax
- **Flexible Speaker Functions**: Multiple speaking order strategies (round robin, random, priority-based)
- **Enhanced Collaboration**: Agents build upon each other's responses and avoid redundancy
- **Interactive Sessions**: Support for both automated and interactive conversation modes
- **Context Awareness**: Agents maintain conversation history and context

For comprehensive documentation on Interactive GroupChat, visit: [Interactive GroupChat Documentation](https://docs.swarms.world/en/latest/swarms/structs/interactive_groupchat/)

### Step-by-Step Showcase

* **Agent Creation**: Define specialized agents with unique expertise and system prompts
* **GroupChat Initialization**: Create the GroupChat structure with desired speaker function
* **Task Definition**: Formulate tasks using `@agent_name` mentions to direct specific agents
* **Execution**: Run the group chat to generate collaborative responses
* **Response Processing**: Handle the coordinated output from multiple agents
* **Iteration**: Chain multiple tasks for complex workflows

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

```python
"""
GroupChat Speaker Function Examples

This example demonstrates how to use different speaker functions in the GroupChat:
- Round Robin: Agents speak in a fixed order, cycling through the list
- Random: Agents speak in random order
- Priority: Agents speak based on priority weights
- Custom: User-defined speaker functions

The example also shows how agents can mention each other using @agent_name syntax.
"""

from swarms import Agent
from swarms.structs.groupchat import GroupChat, random_speaker


def create_example_agents():
    """Create example agents for demonstration."""

    # Create agents with different expertise
    analyst = Agent(
        agent_name="analyst",
        system_prompt="You are a data analyst. You excel at analyzing data, creating charts, and providing insights.",
        model_name="gpt-4.1",
        streaming_on=True,
        print_on=True,
    )

    researcher = Agent(
        agent_name="researcher",
        system_prompt="You are a research specialist. You are great at gathering information, fact-checking, and providing detailed research.",
        model_name="gpt-4.1",
        streaming_on=True,
        print_on=True,
    )

    writer = Agent(
        agent_name="writer",
        system_prompt="You are a content writer. You excel at writing clear, engaging content and summarizing information.",
        model_name="gpt-4.1",
        streaming_on=True,
        print_on=True,
    )

    return [analyst, researcher, writer]


def example_random():
    agents = create_example_agents()

    # Create group chat with random speaker function
    group_chat = GroupChat(
        name="Random Team",
        description="A team that speaks in random order",
        agents=agents,
        speaker_function=random_speaker,
        interactive=False,
    )

    # Test the random behavior
    task = "Let's create a marketing strategy. @analyst @researcher @writer please contribute."

    response = group_chat.run(task)
    print(f"Response:\n{response}\n")


if __name__ == "__main__":
    # example_round_robin()
    example_random()
```



## Connect With Us

Join our community of agent engineers and researchers for technical support, cutting-edge updates, and exclusive access to world-class agent engineering insights!

| Platform | Description | Link |
|----------|-------------|------|
| 📚 Documentation | Official documentation and guides | [docs.swarms.world](https://docs.swarms.world) |
| 📝 Blog | Latest updates and technical articles | [Medium](https://medium.com/@kyeg) |
| 💬 Discord | Live chat and community support | [Join Discord](https://discord.gg/EamjgSaEQf) |
| 🐦 Twitter | Latest news and announcements | [@kyegomez](https://twitter.com/kyegomez) |
| 👥 LinkedIn | Professional network and updates | [The Swarm Corporation](https://www.linkedin.com/company/the-swarm-corporation) |
| 📺 YouTube | Tutorials and demos | [Swarms Channel](https://www.youtube.com/channel/UC9yXyitkbU_WSy7bd_41SqQ) |
| 🎫 Events | Join our community events | [Sign up here](https://lu.ma/swarms_calendar) |
| 🚀 Onboarding Session | Get onboarded with Kye Gomez, creator and lead maintainer of Swarms | [Book Session](https://cal.com/swarms/swarms-onboarding-session) |
