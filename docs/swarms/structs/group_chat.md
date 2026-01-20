# GroupChat Documentation

The GroupChat is a sophisticated multi-agent system that enables interactive conversations between users and AI agents using @mentions. This system allows users to direct tasks to specific agents and facilitates collaborative responses when multiple agents are mentioned.

## Features

| Feature | Description |
|---------|-------------|
| **@mentions Support** | Direct tasks to specific agents using @agent_name syntax |
| **Multi-Agent Collaboration** | Multiple mentioned agents can see and respond to each other's tasks |
| **Enhanced Collaborative Prompts** | Agents are trained to acknowledge, build upon, and synthesize each other's responses |
| **Speaker Functions** | Control the order in which agents respond (round robin, random, priority, custom) |
| **Dynamic Speaker Management** | Change speaker functions and priorities during runtime |
| **Random Dynamic Speaker** | Advanced speaker function that follows @mentions in agent responses |
| **Parallel and Sequential Strategies** | Support for both parallel and sequential agent execution |
| **Callable Function Support** | Supports both Agent instances and callable functions as chat participants |
| **Comprehensive Error Handling** | Custom error classes for different scenarios |
| **Conversation History** | Maintains a complete history of the conversation |
| **Flexible Output Formatting** | Configurable output format for conversation history |
| **Interactive Terminal Mode** | Full REPL interface for real-time chat with agents |

## Installation

```bash
pip install swarms
```

## Methods Reference

### Constructor (`__init__`)

**Description:**
Initializes a new GroupChat instance with the specified configuration.

**Arguments:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `id` | str | Unique identifier for the chat | auto-generated key |
| `name` | str | Name of the group chat | "GroupChat" |
| `description` | str | Description of the chat's purpose | generic description |
| `agents` | List[Union[Agent, Callable]] | List of participating agents | empty list |
| `max_loops` | int | Maximum conversation turns | 1 |
| `output_type` | str | Type of output format | "string" |
| `interactive` | bool | Whether to enable interactive mode | False |
| `speaker_function` | Union[str, Callable] | Function to determine speaking order | round_robin_speaker |
| `speaker_state` | dict | Initial state for speaker function | {"current_index": 0} |

**Example:**

if __name__ == "__main__":

    # Example agents
    agent1 = Agent(
        agent_name="Financial-Analysis-Agent",
        system_prompt="You are a financial analyst specializing in investment strategies.",
        model_name="gpt-4.1",
        max_loops=1,
    )

    agent2 = Agent(
        agent_name="Tax-Adviser-Agent",
        system_prompt="You are a tax adviser who provides clear and concise guidance on tax-related queries.",
        model_name="gpt-4.1",
        max_loops=1,
    )

    agents = [agent1, agent2]

    chat = GroupChat(
        name="Investment Advisory",
        description="Financial and tax analysis group",
        agents=agents,
        speaker_fn=expertise_based,
    )

    history = chat.run(
        "How to optimize tax strategy for investments?"
    )
    print(history.model_dump_json(indent=2))

```

## Speaker Functions


### Built-in Functions

```python
def round_robin(history: List[str], agent: Agent) -> bool:
    """
    Enables agents to speak in turns.
    Returns True for each agent in sequence.
    """
    return True

def expertise_based(history: List[str], agent: Agent) -> bool:
    """
    Enables agents to speak based on their expertise.
    Returns True if agent's role matches conversation context.
    """
    return agent.system_prompt.lower() in history[-1].lower() if history else True

def random_selection(history: List[str], agent: Agent) -> bool:
    """
    Randomly selects speaking agents.
    Returns True/False with 50% probability.
    """
    import random
    return random.choice([True, False])

def most_recent(history: List[str], agent: Agent) -> bool:
    """
    Enables agents to respond to their mentions.
    Returns True if agent was last speaker.
    """
    return agent.agent_name == history[-1].split(":")[0].strip() if history else True
```

### Custom Speaker Function Example

```python
def custom_speaker(history: List[str], agent: Agent) -> bool:
    """
    Custom speaker function with complex logic.
    
    Args:
        history: Previous conversation messages
        agent: Current agent being evaluated
        
    Returns:
        bool: Whether agent should speak
    """
    # No history - let everyone speak
    if not history:
        return True
        
    last_message = history[-1].lower()
    
    # Check for agent expertise keywords
    expertise_relevant = any(
        keyword in last_message 
        for keyword in agent.expertise_keywords
    )
    
    # Check for direct mentions
    mentioned = agent.agent_name.lower() in last_message
    
    # Check if agent hasn't spoken recently
    not_recent_speaker = not any(
        agent.agent_name in msg 
        for msg in history[-3:]
    )
    
    return expertise_relevant or mentioned or not_recent_speaker

# Usage
chat = GroupChat(
    agents=[agent1, agent2],
    speaker_fn=custom_speaker
)
```

## Advanced Examples

### Multi-Agent Analysis Team

```python
# Create specialized agents
data_analyst = Agent(
    agent_name="Data-Analyst",
    system_prompt="You analyze numerical data and patterns",
    model_name="gpt-4.1",
)

market_expert = Agent(
    agent_name="Market-Expert",
    system_prompt="You provide market insights and trends",
    model_name="gpt-4.1",
)

strategy_advisor = Agent(
    agent_name="Strategy-Advisor",
    system_prompt="You formulate strategic recommendations",
    model_name="gpt-4.1",
)

# Create analysis team
analysis_team = GroupChat(
    name="Market Analysis Team",
    description="Comprehensive market analysis group",
    agents=[data_analyst, market_expert, strategy_advisor],
    speaker_fn=expertise_based,
    max_loops=15
)

# Run complex analysis
history = analysis_team.run("""
    Analyze the current market conditions:
    1. Identify key trends
    2. Evaluate risks
    3. Recommend investment strategy
""")
```

### Parallel Processing

```python
# Define multiple analysis tasks
tasks = [
    "Analyze tech sector trends",
    "Evaluate real estate market",
    "Review commodity prices",
    "Assess global economic indicators"
]

# Run tasks concurrently
histories = chat.concurrent_run(tasks)

# Process results
for task, history in zip(tasks, histories):
    print(f"\nAnalysis for: {task}")
    for turn in history.turns:
        for response in turn.responses:
            print(f"{response.agent_name}: {response.message}")
```

## Best Practices

| Category            | Recommendations                                                                                   |
|---------------------|--------------------------------------------------------------------------------------------------|
| **Agent Design**    | - Give agents clear, specific roles<br>- Use detailed system prompts<br>- Set appropriate context lengths<br>- Enable retries for reliability |
| **Speaker Functions** | - Match function to use case<br>- Consider conversation flow<br>- Handle edge cases<br>- Add appropriate logging |
| **Error Handling**  | - Use try-except blocks<br>- Log errors appropriately<br>- Implement retry logic<br>- Provide fallback responses |
| **Performance**     | - Use concurrent processing for multiple tasks<br>- Monitor context lengths<br>- Implement proper cleanup<br>- Cache responses when appropriate |

## API Reference

### GroupChat Methods

| Method | Description | Arguments | Returns |
|--------|-------------|-----------|---------|
| run | Run single conversation | task: str | ChatHistory |
| batched_run | Run multiple sequential tasks | tasks: List[str] | List[ChatHistory] |
| concurrent_run | Run multiple parallel tasks | tasks: List[str] | List[ChatHistory] |
| get_recent_messages | Get recent messages | n: int = 3 | List[str] |

### Agent Methods

| Method | Description | Returns |
|--------|-------------|---------|
| run | Process single task | str |
| generate_response | Generate LLM response | str |
| save_context | Save conversation context | None |