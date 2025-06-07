# InteractiveGroupChat Documentation

The InteractiveGroupChat is a sophisticated multi-agent system that enables interactive conversations between users and AI agents using @mentions. This system allows users to direct messages to specific agents and facilitates collaborative responses when multiple agents are mentioned.

## Features

- **@mentions Support**: Direct messages to specific agents using @agent_name syntax
- **Multi-Agent Collaboration**: Multiple mentioned agents can see and respond to each other's messages
- **Callable Function Support**: Supports both Agent instances and callable functions as chat participants
- **Comprehensive Error Handling**: Custom error classes for different scenarios
- **Conversation History**: Maintains a complete history of the conversation
- **Flexible Output Formatting**: Configurable output format for conversation history

## Installation

```bash
pip install swarms
```

## Basic Usage

```python
from swarms import Agent, InteractiveGroupChat

# Initialize agents
financial_advisor = Agent(
    agent_name="FinancialAdvisor",
    system_prompt="You are a financial advisor specializing in investment strategies.",
    model_name="gpt-4o-mini"
)

tax_expert = Agent(
    agent_name="TaxExpert",
    system_prompt="You are a tax expert providing tax-related guidance.",
    model_name="gpt-4o-mini"
)

# Create the interactive group chat
chat = InteractiveGroupChat(
    name="Financial Team",
    description="Financial advisory team",
    agents=[financial_advisor, tax_expert]
)

# Send a message to a single agent
response = chat.run("@FinancialAdvisor what are good investment strategies?")

# Send a message to multiple agents
response = chat.run("@FinancialAdvisor and @TaxExpert, how can I optimize my investment taxes?")
```

## Advanced Usage

### Using Callable Functions

```python
def custom_agent(context: str) -> str:
    """A custom callable function that can act as an agent"""
    return "Custom response based on: " + context

# Add both Agent instances and callable functions
agents = [financial_advisor, tax_expert, custom_agent]
chat = InteractiveGroupChat(agents=agents)

# Interact with the callable function
response = chat.run("@custom_agent what do you think?")
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| name | str | "InteractiveGroupChat" | Name of the group chat |
| description | str | "An interactive group chat..." | Description of the chat's purpose |
| agents | List[Union[Agent, Callable]] | [] | List of agents or callable functions |
| max_loops | int | 1 | Maximum conversation turns |
| output_type | str | "string" | Output format type |

## Error Handling

The system includes several custom error classes:

- **InteractiveGroupChatError**: Base exception class
- **AgentNotFoundError**: Raised when a mentioned agent doesn't exist
- **NoMentionedAgentsError**: Raised when no agents are mentioned
- **InvalidMessageFormatError**: Raised for invalid message formats

Example error handling:

```python
try:
    response = chat.run("@NonExistentAgent hello!")
except AgentNotFoundError as e:
    print(f"Agent not found: {e}")
except NoMentionedAgentsError as e:
    print(f"No agents mentioned: {e}")
```

## Best Practices

1. **Agent Naming**: Use clear, unique names for agents to avoid confusion
2. **Message Format**: Always use @mentions to direct messages to specific agents
3. **Error Handling**: Implement proper error handling for various scenarios
4. **Context Management**: Be aware that agents can see the full conversation history
5. **Resource Management**: Consider the number of agents and message length to optimize performance

## Logging

The system uses loguru for comprehensive logging:

```python
from loguru import logger

# Configure logging
logger.add("groupchat.log", rotation="500 MB")

# Logs will include:
# - Agent responses
# - Error messages
# - System events
```

## Examples

### Basic Interaction

```python
# Single agent interaction
response = chat.run("@FinancialAdvisor what are the best investment strategies for 2024?")

# Multiple agent collaboration
response = chat.run("@TaxExpert and @InvestmentAnalyst, how can we optimize investment taxes?")
```

### Error Handling

```python
try:
    # Invalid agent mention
    response = chat.run("@NonExistentAgent hello!")
except AgentNotFoundError as e:
    print(f"Error: {e}")

try:
    # No mentions
    response = chat.run("Hello everyone!")
except NoMentionedAgentsError as e:
    print(f"Error: {e}")
```

### Custom Callable Integration

```python
def market_analyzer(context: str) -> str:
    """Custom market analysis function"""
    return "Market analysis based on: " + context

agents = [financial_advisor, tax_expert, market_analyzer]
chat = InteractiveGroupChat(agents=agents)

response = chat.run("@market_analyzer what's your analysis of the current market?")
```

## API Reference

### InteractiveGroupChat Class

```python
class InteractiveGroupChat:
    def __init__(
        self,
        name: str = "InteractiveGroupChat",
        description: str = "An interactive group chat for multiple agents",
        agents: List[Union[Agent, Callable]] = [],
        max_loops: int = 1,
        output_type: str = "string",
    ):
        """Initialize the interactive group chat."""
        
    def run(self, message: str) -> str:
        """Process a message and get responses from mentioned agents."""
```

### Custom Error Classes

```python
class InteractiveGroupChatError(Exception):
    """Base exception class for InteractiveGroupChat errors"""

class AgentNotFoundError(InteractiveGroupChatError):
    """Raised when a mentioned agent is not found"""

class NoMentionedAgentsError(InteractiveGroupChatError):
    """Raised when no agents are mentioned"""

class InvalidMessageFormatError(InteractiveGroupChatError):
    """Raised when the message format is invalid"""
```

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to our GitHub repository.

## License

This project is licensed under the Apache License - see the LICENSE file for details. 