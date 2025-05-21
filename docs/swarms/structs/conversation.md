# Module/Class Name: Conversation

## Introduction

The `Conversation` class is a powerful tool for managing and structuring conversation data in a Python program. It enables you to create, manipulate, and analyze conversations easily. This documentation will provide you with a comprehensive understanding of the `Conversation` class, its attributes, methods, and how to effectively use it.

## Table of Contents

1. **Class Definition**

  - Overview
  
  - Attributes
  
  - Initialization Parameters

2. **Core Methods**

  - Message Management
  
  - History Operations
  
  - Export/Import
  
  - Search and Query
  
  - Cache Management
  
  - Memory Management

3. **Advanced Features**
  
  - Token Counting
  
  - Memory Providers
  
  - Caching System
  
  - Batch Operations

---

### 1. Class Definition

#### Overview

The `Conversation` class is designed to manage conversations by keeping track of messages and their attributes. It offers methods for adding, deleting, updating, querying, and displaying messages within the conversation. Additionally, it supports exporting and importing conversations, searching for specific keywords, and more.

#### Attributes

- `id (str)`: Unique identifier for the conversation

- `name (str)`: Name of the conversation

- `system_prompt (Optional[str])`: System prompt for the conversation

- `time_enabled (bool)`: Flag to enable time tracking for messages

- `autosave (bool)`: Flag to enable automatic saving

- `save_filepath (str)`: File path for saving conversation history

- `conversation_history (list)`: List storing conversation messages

- `tokenizer (Any)`: Tokenizer for counting tokens

- `context_length (int)`: Maximum number of tokens allowed

- `rules (str)`: Rules for the conversation

- `custom_rules_prompt (str)`: Custom prompt for rules

- `user (str)`: User identifier for messages

- `auto_save (bool)`: Flag for auto-saving

- `save_as_yaml (bool)`: Flag to save as YAML


- `save_as_json_bool (bool)`: Flag to save as JSON

- `token_count (bool)`: Flag to enable token counting

- `cache_enabled (bool)`: Flag to enable prompt caching

- `cache_stats (dict)`: Statistics about cache usage

- `provider (Literal["mem0", "in-memory"])`: Memory provider type

#### Initialization Parameters

```python
conversation = Conversation(
    id="unique_id",                    # Optional: Unique identifier
    name="conversation_name",          # Optional: Name of conversation
    system_prompt="System message",    # Optional: Initial system prompt
    time_enabled=True,                 # Optional: Enable timestamps
    autosave=True,                     # Optional: Enable auto-saving
    save_filepath="path/to/save.json", # Optional: Save location
    tokenizer=your_tokenizer,          # Optional: Token counter
    context_length=8192,               # Optional: Max tokens
    rules="conversation rules",        # Optional: Rules
    custom_rules_prompt="custom",      # Optional: Custom rules
    user="User:",                      # Optional: User identifier
    auto_save=True,                    # Optional: Auto-save
    save_as_yaml=True,                 # Optional: Save as YAML
    save_as_json_bool=False,           # Optional: Save as JSON
    token_count=True,                  # Optional: Count tokens
    cache_enabled=True,                # Optional: Enable caching
    conversations_dir="path/to/dir",   # Optional: Cache directory
    provider="in-memory"               # Optional: Memory provider
)
```

### 2. Core Methods

#### Message Management

##### `add(role: str, content: Union[str, dict, list], metadata: Optional[dict] = None)`

Adds a message to the conversation history.

```python
# Add a simple text message
conversation.add("user", "Hello, how are you?")

# Add a structured message
conversation.add("assistant", {
    "type": "response",
    "content": "I'm doing well!"
})

# Add with metadata
conversation.add("user", "Hello", metadata={"timestamp": "2024-03-20"})
```

##### `add_multiple_messages(roles: List[str], contents: List[Union[str, dict, list]])`

Adds multiple messages at once.

```python
conversation.add_multiple_messages(
    roles=["user", "assistant"],
    contents=["Hello!", "Hi there!"]
)
```

##### `add_tool_output_to_agent(role: str, tool_output: dict)`

Adds a tool output to the conversation.

```python
conversation.add_tool_output_to_agent(
    "tool",
    {"name": "calculator", "result": "42"}
)
```

#### History Operations

##### `get_last_message_as_string() -> str`

Returns the last message as a string.

```python
last_message = conversation.get_last_message_as_string()
# Returns: "assistant: Hello there!"
```

##### `get_final_message() -> str`

Returns the final message from the conversation.

```python
final_message = conversation.get_final_message()
# Returns: "assistant: Goodbye!"
```

##### `get_final_message_content() -> str`

Returns just the content of the final message.

```python
final_content = conversation.get_final_message_content()
# Returns: "Goodbye!"
```

##### `return_all_except_first() -> list`

Returns all messages except the first one.

```python
messages = conversation.return_all_except_first()
```

##### `return_all_except_first_string() -> str`

Returns all messages except the first one as a string.

```python
messages_str = conversation.return_all_except_first_string()
```

#### Export/Import

##### `to_json() -> str`

Converts conversation to JSON string.

```python
json_str = conversation.to_json()
```

##### `to_dict() -> list`

Converts conversation to dictionary.

```python
dict_data = conversation.to_dict()
```

##### `to_yaml() -> str`

Converts conversation to YAML string.

```python
yaml_str = conversation.to_yaml()
```

##### `return_json() -> str`

Returns conversation as formatted JSON string.

```python
json_str = conversation.return_json()
```

#### Search and Query

##### `get_visible_messages(agent: "Agent", turn: int) -> List[Dict]`

Gets visible messages for a specific agent and turn.

```python
visible_msgs = conversation.get_visible_messages(agent, turn=1)
```

#### Cache Management

##### `get_cache_stats() -> Dict[str, int]`

Gets statistics about cache usage.

```python
stats = conversation.get_cache_stats()
# Returns: {
#     "hits": 10,
#     "misses": 5,
#     "cached_tokens": 1000,
#     "total_tokens": 2000,
#     "hit_rate": 0.67
# }
```

#### Memory Management

##### `clear_memory()`

Clears the conversation memory.

```python
conversation.clear_memory()
```

##### `clear()`

Clears the conversation history.

```python
conversation.clear()
```

### 3. Advanced Features

#### Token Counting

The class supports automatic token counting when enabled:

```python
conversation = Conversation(token_count=True)
conversation.add("user", "Hello world")
# Token count will be automatically calculated and stored
```

#### Memory Providers

The class supports different memory providers:

```python
# In-memory provider (default)
conversation = Conversation(provider="in-memory")

# Mem0 provider
conversation = Conversation(provider="mem0")
```

#### Caching System

The caching system can be enabled to improve performance:

```python
conversation = Conversation(cache_enabled=True)
# Messages will be cached for faster retrieval
```

#### Batch Operations

The class supports batch operations for efficiency:

```python
# Batch add messages
conversation.batch_add([
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi"}
])
```

### Class Methods

#### `load_conversation(name: str, conversations_dir: Optional[str] = None) -> "Conversation"`

Loads a conversation from cache.

```python
conversation = Conversation.load_conversation("my_conversation")
```

#### `list_cached_conversations(conversations_dir: Optional[str] = None) -> List[str]`

Lists all cached conversations.

```python
conversations = Conversation.list_cached_conversations()
```

## Conclusion

The `Conversation` class provides a comprehensive set of tools for managing conversations in Python applications. With support for multiple memory providers, caching, token counting, and various export formats, it's suitable for a wide range of use cases from simple chat applications to complex AI systems.

For more information or specific use cases, please refer to the examples above or consult the source code.
