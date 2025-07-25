# Agent Multi-Agent Communication Methods

The Agent class provides powerful built-in methods for facilitating communication and collaboration between multiple agents. These methods enable agents to talk to each other, pass information, and coordinate complex multi-agent workflows seamlessly.

## Overview

Multi-agent communication is essential for building sophisticated AI systems where different agents need to collaborate, share information, and coordinate their actions. The Agent class provides several methods to facilitate this communication:

| Method | Purpose | Use Case |
|--------|---------|----------|
| `talk_to` | Direct communication between two agents | Agent handoffs, expert consultation |
| `talk_to_multiple_agents` | Concurrent communication with multiple agents | Broadcasting, consensus building |
| `receive_message` | Process incoming messages from other agents | Message handling, task delegation |
| `send_agent_message` | Send formatted messages to other agents | Direct messaging, notifications |

## Features

| Feature                        | Description                                                        |
|---------------------------------|--------------------------------------------------------------------|
| **Direct Agent Communication**  | Enable one-to-one conversations between agents                     |
| **Concurrent Multi-Agent Communication** | Broadcast messages to multiple agents simultaneously         |
| **Message Processing**          | Handle incoming messages with contextual formatting                |
| **Error Handling**              | Robust error handling for failed communications                    |
| **Threading Support**           | Efficient concurrent processing using ThreadPoolExecutor           |
| **Flexible Parameters**         | Support for images, custom arguments, and kwargs                   |

---

## Core Methods

### `talk_to(agent, task, img=None, *args, **kwargs)`

Enables direct communication between the current agent and another agent. The method processes the task, generates a response, and then passes that response to the target agent.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent` | `Any` | Required | The target agent instance to communicate with |
| `task` | `str` | Required | The task or message to send to the agent |
| `img` | `str` | `None` | Optional image path for multimodal communication |
| `*args` | `Any` | - | Additional positional arguments |
| `**kwargs` | `Any` | - | Additional keyword arguments |

**Returns:** `Any` - The response from the target agent

**Usage Example:**

```python
from swarms import Agent

# Create two specialized agents
researcher = Agent(
    agent_name="Research-Agent",
    system_prompt="You are a research specialist focused on gathering and analyzing information.",
    max_loops=1,
)

analyst = Agent(
    agent_name="Analysis-Agent", 
    system_prompt="You are an analytical specialist focused on interpreting research data.",
    max_loops=1,
)

# Agent communication
research_result = researcher.talk_to(
    agent=analyst,
    task="Analyze the market trends for renewable energy stocks"
)

print(research_result)
```

### `talk_to_multiple_agents(agents, task, *args, **kwargs)`

Enables concurrent communication with multiple agents using ThreadPoolExecutor for efficient parallel processing.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `List[Union[Any, Callable]]` | Required | List of agent instances to communicate with |
| `task` | `str` | Required | The task or message to send to all agents |
| `*args` | `Any` | - | Additional positional arguments |
| `**kwargs` | `Any` | - | Additional keyword arguments |

**Returns:** `List[Any]` - List of responses from all agents (or None for failed communications)

**Usage Example:**

```python
from swarms import Agent

# Create multiple specialized agents
agents = [
    Agent(
        agent_name="Financial-Analyst",
        system_prompt="You are a financial analysis expert.",
        max_loops=1,
    ),
    Agent(
        agent_name="Risk-Assessor", 
        system_prompt="You are a risk assessment specialist.",
        max_loops=1,
    ),
    Agent(
        agent_name="Market-Researcher",
        system_prompt="You are a market research expert.",
        max_loops=1,
    )
]

coordinator = Agent(
    agent_name="Coordinator-Agent",
    system_prompt="You coordinate multi-agent analysis.",
    max_loops=1,
)

# Broadcast to multiple agents
responses = coordinator.talk_to_multiple_agents(
    agents=agents,
    task="Evaluate the investment potential of Tesla stock"
)

# Process responses
for i, response in enumerate(responses):
    if response:
        print(f"Agent {i+1} Response: {response}")
    else:
        print(f"Agent {i+1} failed to respond")
```

### `receive_message(agent_name, task, *args, **kwargs)`

Processes incoming messages from other agents with proper context formatting.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_name` | `str` | Required | Name of the sending agent |
| `task` | `str` | Required | The message content |
| `*args` | `Any` | - | Additional positional arguments |
| `**kwargs` | `Any` | - | Additional keyword arguments |

**Returns:** `Any` - The agent's response to the received message

**Usage Example:**

```python
from swarms import Agent

# Create an agent that can receive messages
recipient_agent = Agent(
    agent_name="Support-Agent",
    system_prompt="You provide helpful support and assistance.",
    max_loops=1,
)

# Simulate receiving a message from another agent
response = recipient_agent.receive_message(
    agent_name="Customer-Service-Agent",
    task="A customer is asking about refund policies. Can you help?"
)

print(response)
```

### `send_agent_message(agent_name, message, *args, **kwargs)`

Sends a formatted message from the current agent to a specified target agent.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_name` | `str` | Required | Name of the target agent |
| `message` | `str` | Required | The message to send |
| `*args` | `Any` | - | Additional positional arguments |
| `**kwargs` | `Any` | - | Additional keyword arguments |

**Returns:** `Any` - The result of sending the message

**Usage Example:**

```python
from swarms import Agent

sender_agent = Agent(
    agent_name="Notification-Agent",
    system_prompt="You send notifications and updates.",
    max_loops=1,
)

# Send a message to another agent
result = sender_agent.send_agent_message(
    agent_name="Task-Manager-Agent",
    message="Task XYZ has been completed successfully"
)

print(result)
```


This comprehensive guide covers all aspects of multi-agent communication using the Agent class methods. These methods provide the foundation for building sophisticated multi-agent systems with robust communication capabilities. 