# InteractiveGroupChat Documentation

The InteractiveGroupChat is a sophisticated multi-agent system that enables interactive conversations between users and AI agents using @mentions. This system allows users to direct tasks to specific agents and facilitates collaborative responses when multiple agents are mentioned.

## Features

- **@mentions Support**: Direct tasks to specific agents using @agent_name syntax
- **Multi-Agent Collaboration**: Multiple mentioned agents can see and respond to each other's tasks
- **Callable Function Support**: Supports both Agent instances and callable functions as chat participants
- **Comprehensive Error Handling**: Custom error classes for different scenarios
- **Conversation History**: Maintains a complete history of the conversation
- **Flexible Output Formatting**: Configurable output format for conversation history

## Installation

```bash
pip install swarms
```

## Methods Reference

### Constructor (`__init__`)

**Description:**  
Initializes a new InteractiveGroupChat instance with the specified configuration.

**Arguments:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `id` | str | Unique identifier for the chat | auto-generated key |
| `name` | str | Name of the group chat | "InteractiveGroupChat" |
| `description` | str | Description of the chat's purpose | generic description |
| `agents` | List[Union[Agent, Callable]] | List of participating agents | empty list |
| `max_loops` | int | Maximum conversation turns | 1 |
| `output_type` | str | Type of output format | "string" |
| `interactive` | bool | Whether to enable interactive mode | False |

**Example:**

```python
from swarms import Agent, InteractiveGroupChat

# Create agents
financial_advisor = Agent(
    agent_name="FinancialAdvisor",
    system_prompt="You are a financial advisor specializing in investment strategies.",
    model_name="gpt-4"
)

tax_expert = Agent(
    agent_name="TaxExpert",
    system_prompt="You are a tax expert providing tax-related guidance.",
    model_name="gpt-4"
)

# Initialize group chat
chat = InteractiveGroupChat(
    id="finance-chat-001",
    name="Financial Advisory Team",
    description="Expert financial guidance team",
    agents=[financial_advisor, tax_expert],
    max_loops=3,
    output_type="string",
    interactive=True
)
```

### Run Method (`run`)

**Description:**  
Processes a task and gets responses from mentioned agents. This is the main method for sending tasks in non-interactive mode.

**Arguments:**

- `task` (str): The input task containing @mentions to agents

**Returns:**

- str: Formatted conversation history including agent responses

**Example:**
```python
# Single agent interaction
response = chat.run("@FinancialAdvisor what are the best ETFs for 2024?")
print(response)

# Multiple agent collaboration
response = chat.run("@FinancialAdvisor and @TaxExpert, how can I minimize taxes on my investments?")
print(response)
```

### Start Interactive Session (`start_interactive_session`)

**Description:**  
Starts an interactive terminal session for real-time chat with agents. This creates a REPL (Read-Eval-Print Loop) interface.

**Arguments:**
None

**Example:**

```python
# Initialize chat with interactive mode
chat = InteractiveGroupChat(
    agents=[financial_advisor, tax_expert],
    interactive=True
)

# Start the interactive session
chat.start_interactive_session()
```

### Extract Mentions (`_extract_mentions`)

**Description:**  

Internal method that extracts @mentions from a task. Used by the run method to identify which agents should respond.

**Arguments:**

- `task` (str): The input task to extract mentions from

**Returns:**

- List[str]: List of mentioned agent names

**Example:**
```python
# Internal usage example (not typically called directly)
chat = InteractiveGroupChat(agents=[financial_advisor, tax_expert])
mentions = chat._extract_mentions("@FinancialAdvisor and @TaxExpert, please help")
print(mentions)  # ['FinancialAdvisor', 'TaxExpert']
```

### Validate Initialization (`_validate_initialization`)

**Description:**  

Internal method that validates the group chat configuration during initialization.

**Arguments:**
None

**Example:**

```python
# Internal validation happens automatically during initialization
chat = InteractiveGroupChat(
    agents=[financial_advisor],  # Valid: at least one agent
    max_loops=5  # Valid: positive number
)
```

### Setup Conversation Context (`_setup_conversation_context`)

**Description:**  

Internal method that sets up the initial conversation context with group chat information.

**Arguments:**

None

**Example:**

```python
# Context is automatically set up during initialization
chat = InteractiveGroupChat(
    name="Investment Team",
    description="Expert investment advice",
    agents=[financial_advisor, tax_expert]
)
# The conversation context now includes chat name, description, and agent info
```

### Update Agent Prompts (`_update_agent_prompts`)

**Description:**  

Internal method that updates each agent's system prompt with information about other agents and the group chat.

**Arguments:**

None

**Example:**
```python
# Agent prompts are automatically updated during initialization
chat = InteractiveGroupChat(agents=[financial_advisor, tax_expert])
# Each agent now knows about the other participants in the chat
```

## Error Classes

### InteractiveGroupChatError

**Description:**  

Base exception class for InteractiveGroupChat errors.

**Example:**
```python
try:
    # Some operation that might fail
    chat.run("@InvalidAgent hello")
except InteractiveGroupChatError as e:
    print(f"Chat error occurred: {e}")
```

### AgentNotFoundError

**Description:**  

Raised when a mentioned agent is not found in the group.

**Example:**
```python
try:
    chat.run("@NonExistentAgent hello!")
except AgentNotFoundError as e:
    print(f"Agent not found: {e}")
```

### NoMentionedAgentsError

**Description:**  

Raised when no agents are mentioned in the task.

**Example:**

```python
try:
    chat.run("Hello everyone!")  # No @mentions
except NoMentionedAgentsError as e:
    print(f"No agents mentioned: {e}")
```

### InvalidtaskFormatError

**Description:**  

Raised when the task format is invalid.

**Example:**
```python
try:
    chat.run("@Invalid@Format")
except InvalidtaskFormatError as e:
    print(f"Invalid task format: {e}")
```

## Best Practices

| Best Practice | Description | Example |
|--------------|-------------|---------|
| Agent Naming | Use clear, unique names for agents to avoid confusion | `financial_advisor`, `tax_expert` |
| task Format | Always use @mentions to direct tasks to specific agents | `@financial_advisor What's your investment advice?` |
| Error Handling | Implement proper error handling for various scenarios | `try/except` blocks for `AgentNotFoundError` |
| Context Management | Be aware that agents can see the full conversation history | Monitor conversation length and relevance |
| Resource Management | Consider the number of agents and task length to optimize performance | Limit max_loops and task size |

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to our GitHub repository.

## License

This project is licensed under the Apache License - see the LICENSE file for details. 