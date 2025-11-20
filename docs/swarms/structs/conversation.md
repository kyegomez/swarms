# Module/Class Name: `Conversation`

The `Conversation` class is a powerful and flexible tool for managing agent conversation context. It provides a comprehensive solution for storing, retrieving, and analyzing conversations with in-memory storage, token tracking, and advanced metadata management.

### Key Features

| Feature Category            | Features / Description                                                                                      |
|----------------------------|-------------------------------------------------------------------------------------------------------------|
| **In-Memory Storage**       | - Fast, efficient in-memory storage for conversation history<br>- No external dependencies required<br>- Perfect for development, testing, and single-session applications |
| **Token Management**        | - Built-in token counting with configurable models<br>- Automatic token tracking for input/output messages<br>- Token usage analytics and reporting<br>- Context length management |
| **Metadata and Categories** | - Support for message metadata<br>- Message categorization (input/output)<br>- Role-based message tracking<br>- Custom message IDs |
| **Data Export/Import**      | - JSON and YAML export formats<br>- Automatic saving and loading<br>- Conversation history management<br>- Batch operations support |
| **Advanced Features**       | - Message search and filtering<br>- Conversation analytics<br>- Multi-agent support<br>- Error handling and fallbacks<br>- Type hints and validation |


## 1. Class Definition

### Overview

The `Conversation` class is designed to manage conversations by keeping track of messages and their attributes. It offers methods for adding, deleting, updating, querying, and displaying messages within the conversation. Additionally, it supports exporting and importing conversations, searching for specific keywords, and more.

The class uses **in-memory storage** for fast and efficient conversation management, making it perfect for development, testing, and single-session applications. No external dependencies are required, making it easy to set up and use.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| id | str | Unique identifier for the conversation |
| name | str | Name of the conversation |
| system_prompt | Optional[str] | System prompt for the conversation |
| time_enabled | bool | Flag to enable time tracking for messages |
| autosave | bool | Flag to enable automatic saving |
| save_filepath | str | File path for saving conversation history |
| load_filepath | str | File path for loading conversation history |
| conversation_history | list | List storing conversation messages |
| context_length | int | Maximum tokens allowed in conversation |
| rules | str | Rules for the conversation |
| custom_rules_prompt | str | Custom prompt for rules |
| user | str | User identifier for messages |
| save_as_yaml_on | bool | Flag to save as YAML |
| save_as_json_bool | bool | Flag to save as JSON |
| token_count | bool | Flag to enable token counting |
| message_id_on | bool | Flag to enable message IDs |
| tokenizer_model_name | str | Model name for tokenization |
| conversations_dir | str | Directory to store conversations |
| export_method | str | Export format ("json" or "yaml") |
| dynamic_context_window | bool | Enable dynamic context window management |
| caching | bool | Enable caching features |

## 2. Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| id | str | generated | Unique conversation ID |
| name | str | "conversation-test" | Name of the conversation |
| system_prompt | Optional[str] | None | System prompt for the conversation |
| time_enabled | bool | False | Enable time tracking |
| autosave | bool | False | Enable automatic saving |
| save_filepath | str | None | File path for saving |
| load_filepath | str | None | File path for loading |
| context_length | int | 8192 | Maximum tokens allowed |
| rules | str | None | Conversation rules |
| custom_rules_prompt | str | None | Custom rules prompt |
| user | str | "User" | User identifier |
| save_as_yaml_on | bool | False | Save as YAML |
| save_as_json_bool | bool | False | Save as JSON |
| token_count | bool | False | Enable token counting |
| message_id_on | bool | False | Enable message IDs |
| tokenizer_model_name | str | "gpt-4.1" | Model name for tokenization |
| conversations_dir | Optional[str] | None | Directory for conversations |
| export_method | str | "json" | Export format ("json" or "yaml") |
| dynamic_context_window | bool | True | Enable dynamic context window management |
| caching | bool | True | Enable caching features |


## 3. Methods

### `add(role: str, content: Union[str, dict, list], metadata: Optional[dict] = None)`

Adds a message to the conversation history.

| Parameter | Type | Description |
|-----------|------|-------------|
| role | str | Role of the speaker |
| content | Union[str, dict, list] | Message content |
| metadata | Optional[dict] | Additional metadata |

Example:
```python
conversation = Conversation()
conversation.add("user", "Hello, how are you?")
conversation.add("assistant", "I'm doing well, thank you!")
```

### `add_multiple_messages(roles: List[str], contents: List[Union[str, dict, list]])`

Adds multiple messages to the conversation history.

| Parameter | Type | Description |
|-----------|------|-------------|
| roles | List[str] | List of speaker roles |
| contents | List[Union[str, dict, list]] | List of message contents |

Example:
```python
conversation = Conversation()
conversation.add_multiple_messages(
    ["user", "assistant"],
    ["Hello!", "Hi there!"]
)
```

### `delete(index: str)`

Deletes a message from the conversation history.

| Parameter | Type | Description |
|-----------|------|-------------|
| index | str | Index of message to delete |

Example:
```python
conversation = Conversation()
conversation.add("user", "Hello")
conversation.delete(0)  # Deletes the first message
```

### `update(index: str, role: str, content: Union[str, dict])`

Updates a message in the conversation history.

| Parameter | Type | Description |
|-----------|------|-------------|
| index | str | Index of message to update |
| role | str | New role of speaker |
| content | Union[str, dict] | New message content |

Example:
```python
conversation = Conversation()
conversation.add("user", "Hello")
conversation.update(0, "user", "Hi there!")
```

### `query(index: str)`

Retrieves a message from the conversation history.

| Parameter | Type | Description |
|-----------|------|-------------|
| index | str | Index of message to query |

Example:
```python
conversation = Conversation()
conversation.add("user", "Hello")
message = conversation.query(0)
```

### `search(keyword: str)`

Searches for messages containing a keyword.

| Parameter | Type | Description |
|-----------|------|-------------|
| keyword | str | Keyword to search for |

Example:
```python
conversation = Conversation()
conversation.add("user", "Hello world")
results = conversation.search("world")
```

### `display_conversation(detailed: bool = False)`

Displays the conversation history.

| Parameter | Type | Description |
|-----------|------|-------------|
| detailed | bool | Show detailed information |

Example:
```python
conversation = Conversation()
conversation.add("user", "Hello")
conversation.display_conversation(detailed=True)
```

### `export_conversation(filename: str)`

Exports conversation history to a file.

| Parameter | Type | Description |
|-----------|------|-------------|
| filename | str | Output file path |

Example:
```python
conversation = Conversation()
conversation.add("user", "Hello")
conversation.export_conversation("chat.txt")
```

### `import_conversation(filename: str)`

Imports conversation history from a file.

| Parameter | Type | Description |
|-----------|------|-------------|
| filename | str | Input file path |

Example:
```python
conversation = Conversation()
conversation.import_conversation("chat.txt")
```

### `count_messages_by_role()`

Counts messages by role.

Returns: Dict[str, int]

Example:
```python
conversation = Conversation()
conversation.add("user", "Hello")
conversation.add("assistant", "Hi")
counts = conversation.count_messages_by_role()
```

### `return_history_as_string()`

Returns conversation history as a string.

Returns: str

Example:
```python
conversation = Conversation()
conversation.add("user", "Hello")
history = conversation.return_history_as_string()
```

### `save_as_json(filename: str)`

Saves conversation history as JSON.

| Parameter | Type | Description |
|-----------|------|-------------|
| filename | str | Output JSON file path |

Example:
```python
conversation = Conversation()
conversation.add("user", "Hello")
conversation.save_as_json("chat.json")
```

### `load_from_json(filename: str)`

Loads conversation history from JSON.

| Parameter | Type | Description |
|-----------|------|-------------|
| filename | str | Input JSON file path |

Example:
```python
conversation = Conversation()
conversation.load_from_json("chat.json")
```

### `truncate_memory_with_tokenizer()`

Truncates conversation history based on token limit.

Example:
```python
conversation = Conversation(tokenizer=some_tokenizer)
conversation.truncate_memory_with_tokenizer()
```

### `clear()`

Clears the conversation history.

Example:
```python
conversation = Conversation()
conversation.add("user", "Hello")
conversation.clear()
```

### `to_json()`

Converts conversation history to JSON string.

Returns: str

Example:
```python
conversation = Conversation()
conversation.add("user", "Hello")
json_str = conversation.to_json()
```

### `to_dict()`

Converts conversation history to dictionary.

Returns: list

Example:
```python
conversation = Conversation()
conversation.add("user", "Hello")
dict_data = conversation.to_dict()
```

### `to_yaml()`

Converts conversation history to YAML string.

Returns: str

Example:
```python
conversation = Conversation()
conversation.add("user", "Hello")
yaml_str = conversation.to_yaml()
```

### `get_visible_messages(agent: "Agent", turn: int)`

Gets visible messages for an agent at a specific turn.

| Parameter | Type | Description |
|-----------|------|-------------|
| agent | Agent | The agent |
| turn | int | Turn number |

Returns: List[Dict]

Example:
```python
conversation = Conversation()
visible_msgs = conversation.get_visible_messages(agent, 1)
```

### `get_last_message_as_string()`

Gets the last message as a string.

Returns: str

Example:
```python
conversation = Conversation()
conversation.add("user", "Hello")
last_msg = conversation.get_last_message_as_string()
```

### `return_messages_as_list()`

Returns messages as a list of strings.

Returns: List[str]

Example:
```python
conversation = Conversation()
conversation.add("user", "Hello")
messages = conversation.return_messages_as_list()
```

### `return_messages_as_dictionary()`

Returns messages as a list of dictionaries.

Returns: List[Dict]

Example:
```python
conversation = Conversation()
conversation.add("user", "Hello")
messages = conversation.return_messages_as_dictionary()
```

### `add_tool_output_to_agent(role: str, tool_output: dict)`

Adds tool output to conversation.

| Parameter | Type | Description |
|-----------|------|-------------|
| role | str | Role of the tool |
| tool_output | dict | Tool output to add |

Example:
```python
conversation = Conversation()
conversation.add_tool_output_to_agent("tool", {"result": "success"})
```

### `return_json()`

Returns conversation as JSON string.

Returns: str

Example:
```python
conversation = Conversation()
conversation.add("user", "Hello")
json_str = conversation.return_json()
```

### `get_final_message()`

Gets the final message.

Returns: str

Example:
```python
conversation = Conversation()
conversation.add("user", "Hello")
final_msg = conversation.get_final_message()
```

### `get_final_message_content()`

Gets content of final message.

Returns: str

Example:
```python
conversation = Conversation()
conversation.add("user", "Hello")
content = conversation.get_final_message_content()
```

### `return_all_except_first()`

Returns all messages except first.

Returns: List[Dict]

Example:
```python
conversation = Conversation()
conversation.add("system", "Start")
conversation.add("user", "Hello")
messages = conversation.return_all_except_first()
```

### `return_all_except_first_string()`

Returns all messages except first as string.

Returns: str

Example:
```python
conversation = Conversation()
conversation.add("system", "Start")
conversation.add("user", "Hello")
messages = conversation.return_all_except_first_string()
```

### `batch_add(messages: List[dict])`

Adds multiple messages in batch.

| Parameter | Type | Description |
|-----------|------|-------------|
| messages | List[dict] | List of messages to add |

Example:
```python
conversation = Conversation()
conversation.batch_add([
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi"}
])
```

### `get_cache_stats()`

Gets cache usage statistics.

Returns: Dict[str, int]

Example:
```python
conversation = Conversation()
stats = conversation.get_cache_stats()
```

### `load_conversation(name: str, conversations_dir: Optional[str] = None)`

Loads a conversation from cache.

| Parameter | Type | Description |
|-----------|------|-------------|
| name | str | Name of conversation |
| conversations_dir | Optional[str] | Directory containing conversations |

Returns: Conversation

Example:
```python
conversation = Conversation.load_conversation("my_chat")
```

### `list_cached_conversations(conversations_dir: Optional[str] = None)`

Lists all cached conversations.

| Parameter | Type | Description |
|-----------|------|-------------|
| conversations_dir | Optional[str] | Directory containing conversations |

Returns: List[str]

Example:
```python
conversations = Conversation.list_cached_conversations()
```

### `clear_memory()`

Clears the conversation memory.

Example:
```python
conversation = Conversation()
conversation.add("user", "Hello")
conversation.clear_memory()
```

## 4. Examples

### Basic Usage

```python
from swarms.structs import Conversation

# Create a new conversation with in-memory storage
conversation = Conversation(
    name="my_chat",
    system_prompt="You are a helpful assistant",
    time_enabled=True,
    token_count=True,
    tokenizer_model_name="gpt-4.1",
    message_id_on=True,
    export_method="json",
    autosave=True
)

# Add messages with metadata and categories
conversation.add(
    role="user", 
    content="Hello!",
    metadata={"session_id": "123"},
    category="input"
)

conversation.add(
    role="assistant", 
    content="Hi there!",
    metadata={"response_time": "0.5s"},
    category="output"
)

# Get token usage statistics
token_stats = conversation.export_and_count_categories()
print(f"Input tokens: {token_stats['input_tokens']}")
print(f"Output tokens: {token_stats['output_tokens']}")
print(f"Total tokens: {token_stats['total_tokens']}")

# Display conversation
conversation.display_conversation()
```

### Advanced Configuration with Export/Import

```python
from swarms.structs import Conversation
import os

# Advanced configuration with custom settings
conversation = Conversation(
    name="advanced_chat",
    system_prompt="You are a helpful assistant",
    time_enabled=True,
    token_count=True,
    message_id_on=True,
    export_method="yaml",
    dynamic_context_window=True,
    caching=True,
    autosave=True,
    conversations_dir=os.path.expanduser("~/conversations")
)

# Add structured messages
conversation.add(
    role="user",
    content={
        "message": "Process this data",
        "data": {"key": "value"}
    }
)

# Batch add multiple messages
conversation.batch_add([
    {"role": "assistant", "content": "Processing..."},
    {"role": "system", "content": "Data processed successfully"}
])

# Export conversation to YAML
conversation.export(force=True)  # force=True overrides autosave setting
```

### File-based Persistence

```python
from swarms.structs import Conversation
import os

# Create conversation with file persistence
conversation = Conversation(
    name="persistent_chat",
    system_prompt="You are a helpful assistant",
    token_count=True,
    export_method="json",
    autosave=True,
    save_filepath="my_conversation.json"
)

# Add messages
conversation.add("user", "Hello!")
conversation.add("assistant", "Hi there!")

# Conversation is automatically saved to file
# You can also manually export
conversation.export()

# Load conversation later
loaded_conversation = Conversation.load_conversation("persistent_chat")
```

### Advanced Usage with Multi-Agent Systems

```python
from swarms.structs import Agent, Conversation
from swarms.structs.multi_agent_exec import run_agents_concurrently

# Set up conversation for multi-agent analytics
conversation = Conversation(
    name="multi_agent_analytics",
    system_prompt="Multi-agent analytics session",
    time_enabled=True,
    token_count=True,
    message_id_on=True,
    export_method="json",
    autosave=True
)

# Create specialized agents
data_analyst = Agent(
    agent_name="DataAnalyst",
    system_prompt="You are a senior data analyst...",
    model_name="gpt-4.1",
    max_loops=1
)

researcher = Agent(
    agent_name="ResearchSpecialist", 
    system_prompt="You are a research specialist...",
    model_name="gpt-4.1",
    max_loops=1
)

# Run agents with structured metadata
task = "Analyze the current state of AI in healthcare"
results = run_agents_concurrently(
    agents=[data_analyst, researcher], 
    task=task
)

# Store results with metadata
for result, agent in zip(results, [data_analyst, researcher]):
    conversation.add(
        content=result, 
        role=agent.agent_name,
        metadata={
            "agent_type": agent.agent_name,
            "model": agent.model_name,
            "task": task
        }
    )

# Get analytics
token_usage = conversation.export_and_count_categories()
message_counts = conversation.count_messages_by_role()

# Export conversation for persistence
conversation.export()
```

### Error Handling and Type Hints

```python
from typing import Optional, Dict, Any
from swarms.structs import Conversation

def initialize_conversation(
    name: str,
    config: Dict[str, Any]
) -> Optional[Conversation]:
    """Initialize conversation with error handling."""
    try:
        conversation = Conversation(
            name=name,
            **config
        )
        print(f"✅ Conversation '{name}' initialized successfully")
        return conversation
    except Exception as e:
        print(f"❌ Error initializing conversation: {e}")
        return None

# Usage
config = {
    "system_prompt": "You are a helpful assistant",
    "time_enabled": True,
    "token_count": True,
    "export_method": "json",
    "autosave": True
}

conversation = initialize_conversation(
    name="error_handling_test",
    config=config
)

if conversation:
    conversation.add("user", "Hello!")
    conversation.add("assistant", "Hi there!")
    
    # Export for persistence
    conversation.export()
```

### Loading and Managing Conversations

```python
from swarms.structs import Conversation
from typing import List, Dict
import os

def manage_conversations(base_dir: str) -> List[Dict[str, str]]:
    """Manage conversations with file-based persistence."""
    
    # List all saved conversations
    conversations = Conversation.list_conversations(
        conversations_dir=base_dir
    )
    
    # Print conversation stats
    for conv in conversations:
        print(f"ID: {conv['id']}")
        print(f"Name: {conv['name']}")
        print(f"Created: {conv['created_at']}")
        print(f"Path: {conv['filepath']}")
        print("---")
    
    # Load specific conversation
    if conversations:
        latest = conversations[0]  # Most recent conversation
        conversation = Conversation.load_conversation(
            name=latest["name"],
            conversations_dir=base_dir,
            load_filepath=latest["filepath"]
        )
        
        # Get conversation statistics
        stats = {
            "messages": len(conversation.conversation_history),
            "roles": conversation.count_messages_by_role(),
            "tokens": conversation.export_and_count_categories()
        }
        
        return stats
    
    return []

# Usage
base_dir = os.path.expanduser("~/conversation_data")
stats = manage_conversations(base_dir)
```

### Configuration Examples

```python
# Basic configuration: Simple in-memory storage
conv_basic = Conversation(
    name="basic_chat",
    token_count=True,
    message_id_on=True
)

# Development configuration: With time tracking and autosave
conv_dev = Conversation(
    name="dev_chat",
    time_enabled=True,
    token_count=True,
    message_id_on=True,
    autosave=True,
    export_method="json"
)

# Production configuration: Full features with file persistence
conv_prod = Conversation(
    name="prod_chat",
    system_prompt="You are a helpful assistant",
    time_enabled=True,
    token_count=True,
    message_id_on=True,
    autosave=True,
    export_method="json",
    dynamic_context_window=True,
    caching=True,
    conversations_dir="/app/conversations"
)

# Analytics configuration: For data analysis and reporting
conv_analytics = Conversation(
    name="analytics_chat",
    token_count=True,
    export_method="yaml",
    autosave=True,
    dynamic_context_window=True
)
```

## Error Handling

The conversation class provides graceful error handling:

- **File Operations**: Clear error messages for file read/write issues
- **Data Validation**: Input validation and type checking
- **Memory Management**: Efficient memory usage and cleanup
- **Data Corruption**: Validation and recovery mechanisms

Example error handling:
```python
try:
    conversation = Conversation(name="test")
    conversation.add("user", "Hello")
    conversation.export()
except Exception as e:
    print(f"Error: {e}")
```


## Conclusion

The `Conversation` class provides a comprehensive set of tools for managing conversations in Python applications with efficient in-memory storage. It supports token counting, caching, metadata management, and multiple export/import formats. The class is designed to be simple, fast, and reliable, making it suitable for a wide range of use cases from simple chat applications to complex conversational AI systems.

Key benefits:

| Benefit                  | Description                                      |
|--------------------------|--------------------------------------------------|
| **Simple Setup**         | No external dependencies required                |
| **Fast Performance**     | In-memory storage for quick access               |
| **File Persistence**     | Export/import for data persistence               |
| **Token Management**     | Built-in token counting and analytics            |
| **Flexible Configuration** | Customizable for different use cases           |
| **Type Safety**          | Full type hints and validation                   |
