# GroupChat Class Documentation


The GroupChat class manages multi-agent conversations with state persistence, comprehensive logging, and flexible agent configurations. It supports both Agent class instances and callable functions, making it versatile for different use cases.

## Installation
```bash
pip install swarms python-dotenv pydantic
```


## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| state_path | str | Path for saving/loading chat state |
| wrapped_agents | List[AgentWrapper] | List of wrapped agent instances |
| selector_agent | AgentWrapper | Agent responsible for speaker selection |
| state | GroupChatState | Current state of the group chat |

## Methods

### Core Methods

```python
def run(self, task: str) -> str:
    """Execute the group chat conversation"""

def save_state(self) -> None:
    """Save current state to disk"""

@classmethod
def load_state(cls, state_path: str) -> 'GroupChat':
    """Load GroupChat from saved state"""

def get_conversation_summary(self) -> Dict[str, Any]:
    """Return a summary of the conversation"""

def export_conversation(self, format: str = "json") -> Union[str, Dict]:
    """Export the conversation in specified format"""
```

### Internal Methods

```python
def _log_interaction(self, agent_name: str, position: int, input_text: str, output_text: str) -> None:
    """Log a single interaction"""

def _add_message(self, role: str, content: str) -> None:
    """Add a message to the conversation history"""

def select_next_speaker(self, last_speaker: AgentWrapper) -> AgentWrapper:
    """Select the next speaker using the selector agent"""
```

## Usage Examples

### 1. Basic Setup with Two Agents
```python
import os
from swarms import Agent
from swarm_models import OpenAIChat

# Initialize OpenAI
api_key = os.getenv("OPENAI_API_KEY")
model = OpenAIChat(openai_api_key=api_key, model_name="gpt-4-mini")

# Create agents
analyst = Agent(
    agent_name="Financial-Analyst",
    system_prompt="You are a financial analyst...",
    llm=model
)

advisor = Agent(
    agent_name="Investment-Advisor",
    system_prompt="You are an investment advisor...",
    llm=model
)

# Create group chat
chat = GroupChat(
    name="Investment Team",
    agents=[analyst, advisor],
    max_rounds=5,
    group_objective="Provide investment advice"
)

response = chat.run("What's the best investment strategy for retirement?")
```

### 2. Advanced Setup with State Management
```python
# Create group chat with state persistence
chat = GroupChat(
    name="Investment Advisory Team",
    description="Expert team for financial planning",
    agents=[analyst, advisor, tax_specialist],
    max_rounds=10,
    admin_name="Senior Advisor",
    group_objective="Provide comprehensive financial planning",
    state_path="investment_chat_state.json",
    rules="1. Always provide sources\n2. Be concise\n3. Focus on practical advice"
)

# Run chat and save state
response = chat.run("Create a retirement plan for a 35-year old")
chat.save_state()

# Load existing chat state
loaded_chat = GroupChat.load_state("investment_chat_state.json")
```

### 3. Using Custom Callable Agents
```python
def custom_agent(input_text: str) -> str:
    # Custom logic here
    return f"Processed: {input_text}"

# Mix of regular agents and callable functions
chat = GroupChat(
    name="Hybrid Team",
    agents=[analyst, custom_agent],
    max_rounds=3
)
```

### 4. Export and Analysis
```python
# Run chat
chat.run("Analyze market conditions")

# Get summary
summary = chat.get_conversation_summary()
print(summary)

# Export in different formats
json_conv = chat.export_conversation(format="json")
text_conv = chat.export_conversation(format="text")
```

### 5. Advanced Configuration with Custom Selector
```python
class CustomSelector(Agent):
    def run(self, input_text: str) -> str:
        # Custom selection logic
        return "Financial-Analyst"

chat = GroupChat(
    name="Custom Selection Team",
    agents=[analyst, advisor],
    selector_agent=CustomSelector(
        agent_name="Custom-Selector",
        system_prompt="Select the next speaker based on expertise",
        llm=model
    ),
    max_rounds=5
)
```

### 6. Debugging Setup
```python
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

chat = GroupChat(
    name="Debug Team",
    agents=[analyst, advisor],
    max_rounds=3,
    state_path="debug_chat.json"
)

# Run with detailed logging
try:
    response = chat.run("Complex query")
except Exception as e:
    logger.error(f"Chat failed: {str(e)}")
    # Access last successful state
    state = chat.state
```

## Error Handling

The GroupChat class includes comprehensive error handling:

```python
try:
    chat = GroupChat(agents=[analyst])  # Will raise ValueError
except ValueError as e:
    print("Configuration error:", str(e))

try:
    response = chat.run("Query")
except Exception as e:
    # Access error state
    error_summary = chat.get_conversation_summary()
    print("Execution error:", str(e))
    print("State at error:", error_summary)
```

## Best Practices

1. **State Management**:
   - Always specify a `state_path` for important conversations
   - Use `save_state()` after critical operations
   - Implement regular state backups for long conversations

2. **Agent Configuration**:
   - Provide clear system prompts for each agent
   - Use descriptive agent names
   - Consider agent expertise when setting the group objective

3. **Performance**:
   - Keep `max_rounds` reasonable (5-10 for most cases)
   - Use early stopping conditions when possible
   - Monitor conversation length and complexity

4. **Error Handling**:
   - Always wrap chat execution in try-except blocks
   - Implement proper logging
   - Save states before potentially risky operations

## Limitations

- Agents must either have a `run` method or be callable
- State files can grow large with many interactions
- Selector agent may need optimization for large agent groups
- Real-time streaming not supported in basic configuration

