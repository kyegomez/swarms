# Broadcast Communication Tutorial

A simple guide to using the broadcast communication pattern where one agent sends to many.

## Overview

The **broadcast** function implements a one-to-many communication pattern where a single sender agent broadcasts a message to multiple receiver agents. All receivers process the broadcast message and provide their responses. This is useful for scenarios requiring one central message to be processed by multiple agents in parallel.

| Feature | Description |
|---------|-------------|
| **One-to-Many** | Single sender communicates with multiple receivers |
| **Broadcast Message** | Sender creates a message that all receivers process |
| **Parallel Processing** | All receivers process the broadcast simultaneously |
| **Async Support** | Function is async for concurrent processing |

### When to Use Broadcast

| Scenario | Recommendation |
|----------|----------------|
| Announcing information to multiple agents | Best For |
| Getting multiple perspectives on a single message | Best For |
| Central coordination with distributed processing | Best For |
| Scenarios requiring one source of truth | Best For |
| Two-agent communication (use one_to_one instead) | Not Ideal For |
| Sequential workflows | Not Ideal For |
| Agent-to-agent conversations | Not Ideal For |

---

## Installation

```bash
pip install swarms
```

---

## Quick Start

### Step 1: Import Required Modules

```python
import asyncio
from swarms import Agent
from swarms.structs.swarming_architectures import broadcast
```

### Step 2: Create Your Agents

```python
# Sender agent (broadcasts message)
announcer = Agent(
    agent_name="Announcer",
    system_prompt="You are an announcer. You create clear, informative messages.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Receiver agents (process broadcast)
researcher = Agent(
    agent_name="Researcher",
    system_prompt="You are a researcher. You analyze information.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

analyst = Agent(
    agent_name="Analyst",
    system_prompt="You are an analyst. You provide insights.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

writer = Agent(
    agent_name="Writer",
    system_prompt="You are a writer. You synthesize information.",
    model_name="gpt-4o-mini",
    max_loops=1,
)
```

### Step 3: Define Your Task

```python
task = "Announce the new product launch strategy and gather team input"
```

### Step 4: Run Broadcast Communication

```python
# Execute broadcast (async function)
async def main():
    result = await broadcast(
        sender=announcer,
        agents=[researcher, analyst, writer],
        task=task,
        output_type="dict"
    )
    return result

# Run the async function
result = asyncio.run(main())
print(result)
```

---

## Complete Example

```python
import asyncio
from swarms import Agent
from swarms.structs.swarming_architectures import broadcast

# Step 1: Create sender agent
coordinator = Agent(
    agent_name="Project-Coordinator",
    system_prompt="""You are a project coordinator. You create clear announcements 
    and coordinate team activities.""",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Step 2: Create receiver agents
team_members = [
    Agent(
        agent_name="Developer",
        system_prompt="You are a developer. You evaluate technical feasibility.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="Designer",
        system_prompt="You are a designer. You evaluate design implications.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="Product-Manager",
        system_prompt="You are a product manager. You evaluate product strategy.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
]

# Step 3: Define the task
task = "Announce the new feature requirements and gather team feedback"

# Step 4: Run broadcast
async def main():
    result = await broadcast(
        sender=coordinator,
        agents=team_members,
        task=task,
        output_type="dict"
    )
    
    # Step 5: Access results
    print("Broadcast Results:")
    print(f"Total messages: {len(result.get('messages', []))}\n")
    
    for message in result.get("messages", []):
        role = message['role']
        content = message['content'][:200]
        print(f"{role}:")
        print(f"{content}...\n")
    
    return result

# Execute
result = asyncio.run(main())
```

---

## Understanding the Flow

The broadcast pattern works as follows:

1. **User provides task** to the sender
2. **Sender processes task** and creates a broadcast message
3. **Broadcast message is sent** to all receiver agents
4. **All receivers process** the broadcast message in parallel
5. **All responses are collected** in the conversation history

```
User → Sender → Broadcast Message
                ↓
        [Receiver 1, Receiver 2, Receiver 3, ...]
                ↓
        [Response 1, Response 2, Response 3, ...]
```

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sender` | `Agent` | Required | The agent broadcasting the message |
| `agents` | `AgentListType` | Required | List of agents receiving the broadcast |
| `task` | `str` | Required | The task for the sender to process |
| `output_type` | `OutputType` | `"dict"` | Output format: "dict" or "list" |

---

## Output Types

### Dictionary Output (Default)

```python
async def main():
    result = await broadcast(
        sender=sender,
        agents=[agent1, agent2, agent3],
        task="Your task here",
        output_type="dict"
    )
    return result

result = asyncio.run(main())

# Returns:
# {
#     "messages": [
#         {"role": "User", "content": "Your task here"},
#         {"role": "Announcer", "content": "Broadcast message..."},
#         {"role": "Developer", "content": "Response 1..."},
#         {"role": "Designer", "content": "Response 2..."},
#         {"role": "Product-Manager", "content": "Response 3..."}
#     ]
# }
```

### List Output

```python
async def main():
    result = await broadcast(
        sender=sender,
        agents=[agent1, agent2, agent3],
        task="Your task here",
        output_type="list"
    )
    return result

result = asyncio.run(main())
# Returns list of responses
```

---

## Use Cases

### Use Case 1: Team Announcement

```python
import asyncio
from swarms import Agent
from swarms.structs.swarming_architectures import broadcast

manager = Agent(
    agent_name="Manager",
    system_prompt="You are a manager. You make announcements to your team.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

team = [
    Agent(agent_name=f"Team-Member-{i}", system_prompt="Team member", model_name="gpt-4o-mini", max_loops=1)
    for i in range(5)
]

async def main():
    result = await broadcast(
        sender=manager,
        agents=team,
        task="Announce the new project timeline and gather team concerns",
        output_type="dict"
    )
    return result

result = asyncio.run(main())
```

### Use Case 2: Multi-Perspective Analysis

```python
import asyncio
from swarms import Agent
from swarms.structs.swarming_architectures import broadcast

researcher = Agent(
    agent_name="Lead-Researcher",
    system_prompt="You are a lead researcher. You present research findings.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

analysts = [
    Agent(agent_name="Technical-Analyst", system_prompt="Technical analysis expert", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Business-Analyst", system_prompt="Business analysis expert", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Market-Analyst", system_prompt="Market analysis expert", model_name="gpt-4o-mini", max_loops=1),
]

async def main():
    result = await broadcast(
        sender=researcher,
        agents=analysts,
        task="Present the research findings on AI adoption and get analysis from each perspective",
        output_type="dict"
    )
    return result

result = asyncio.run(main())
```

### Use Case 3: Feedback Collection

```python
import asyncio
from swarms import Agent
from swarms.structs.swarming_architectures import broadcast

presenter = Agent(
    agent_name="Presenter",
    system_prompt="You present proposals clearly and concisely.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

reviewers = [
    Agent(agent_name="Reviewer-1", system_prompt="You provide detailed feedback", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Reviewer-2", system_prompt="You provide detailed feedback", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Reviewer-3", system_prompt="You provide detailed feedback", model_name="gpt-4o-mini", max_loops=1),
]

async def main():
    result = await broadcast(
        sender=presenter,
        agents=reviewers,
        task="Present the new design proposal and collect feedback",
        output_type="dict"
    )
    return result

result = asyncio.run(main())
```

---

## Async/Await Pattern

Since `broadcast` is an async function, you must use `asyncio.run()` or be in an async context:

```python
# Method 1: Using asyncio.run()
import asyncio

result = asyncio.run(broadcast(sender, agents, task))

# Method 2: Inside an async function
async def my_function():
    result = await broadcast(sender, agents, task)
    return result

result = asyncio.run(my_function())

# Method 3: In an async context (e.g., FastAPI endpoint)
@app.post("/broadcast")
async def broadcast_endpoint():
    result = await broadcast(sender, agents, task)
    return result
```

---

## Best Practices

| Practice | Description |
|----------|-------------|
| **Clear Sender Role** | Give the sender a clear broadcasting/announcement role |
| **Receiver Diversity** | Use receivers with different perspectives or expertise |
| **Task Clarity** | Make the task clear for the sender to create a good broadcast |
| **Async Handling** | Remember to use `asyncio.run()` or await in async contexts |

---

## Key Characteristics

| Characteristic | Description |
|---------------|-------------|
| **Async Function** | Must be called with `await` or `asyncio.run()` |
| **One Source** | Single sender creates the broadcast message |
| **Parallel Processing** | All receivers process simultaneously |
| **Full Context** | Receivers see the full conversation including the broadcast |

---

## Related Architectures

| Architecture | When to Use Instead |
|--------------|---------------------|
| **[One-to-One](./one_to_one_example.md)** | For two-agent communication |
| **[Star Swarm](./star_swarm_example.md)** | For central coordination with sequential processing |
| **[Circular Swarm](./circular_swarm_example.md)** | For sequential round-robin processing |
