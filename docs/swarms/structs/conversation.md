# Module/Class Name: Conversation

## Introduction

The `Conversation` class is a powerful tool for managing and structuring conversation data in a Python program. It enables you to create, manipulate, and analyze conversations easily. This documentation provides a comprehensive understanding of the `Conversation` class, its attributes, methods, and how to effectively use it.

## Table of Contents

1. [Class Definition](#1-class-definition)
2. [Initialization Parameters](#2-initialization-parameters)
3. [Methods](#3-methods)
4. [Examples](#4-examples)

## 1. Class Definition

### Overview

The `Conversation` class is designed to manage conversations by keeping track of messages and their attributes. It offers methods for adding, deleting, updating, querying, and displaying messages within the conversation. Additionally, it supports exporting and importing conversations, searching for specific keywords, and more.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| id | str | Unique identifier for the conversation |
| name | str | Name of the conversation |
| system_prompt | Optional[str] | System prompt for the conversation |
| time_enabled | bool | Flag to enable time tracking for messages |
| autosave | bool | Flag to enable automatic saving |
| save_filepath | str | File path for saving conversation history |
| conversation_history | list | List storing conversation messages |
| tokenizer | Any | Tokenizer for counting tokens |
| context_length | int | Maximum tokens allowed in conversation |
| rules | str | Rules for the conversation |
| custom_rules_prompt | str | Custom prompt for rules |
| user | str | User identifier for messages |
| auto_save | bool | Flag to enable auto-saving |
| save_as_yaml | bool | Flag to save as YAML |
| save_as_json_bool | bool | Flag to save as JSON |
| token_count | bool | Flag to enable token counting |
| cache_enabled | bool | Flag to enable prompt caching |
| cache_stats | dict | Statistics about cache usage |
| cache_lock | threading.Lock | Lock for thread-safe cache operations |
| conversations_dir | str | Directory to store cached conversations |

## 2. Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| id | str | generated | Unique conversation ID |
| name | str | None | Name of the conversation |
| system_prompt | Optional[str] | None | System prompt for the conversation |
| time_enabled | bool | False | Enable time tracking |
| autosave | bool | False | Enable automatic saving |
| save_filepath | str | None | File path for saving |
| tokenizer | Any | None | Tokenizer for counting tokens |
| context_length | int | 8192 | Maximum tokens allowed |
| rules | str | None | Conversation rules |
| custom_rules_prompt | str | None | Custom rules prompt |
| user | str | "User:" | User identifier |
| auto_save | bool | True | Enable auto-saving |
| save_as_yaml | bool | True | Save as YAML |
| save_as_json_bool | bool | False | Save as JSON |
| token_count | bool | True | Enable token counting |
| cache_enabled | bool | True | Enable prompt caching |
| conversations_dir | Optional[str] | None | Directory for cached conversations |
| provider | Literal["mem0", "in-memory"] | "in-memory" | Storage provider |

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

# Create a new conversation
conversation = Conversation(
    name="my_chat",
    system_prompt="You are a helpful assistant",
    time_enabled=True
)

# Add messages
conversation.add("user", "Hello!")
conversation.add("assistant", "Hi there!")

# Display conversation
conversation.display_conversation()

# Save conversation
conversation.save_as_json("my_chat.json")
```

### Advanced Usage with Token Counting

```python
from swarms.structs import Conversation
from some_tokenizer import Tokenizer

# Create conversation with token counting
conversation = Conversation(
    tokenizer=Tokenizer(),
    context_length=4096,
    token_count=True
)

# Add messages
conversation.add("user", "Hello, how are you?")
conversation.add("assistant", "I'm doing well, thank you!")

# Get token statistics
stats = conversation.get_cache_stats()
print(f"Total tokens: {stats['total_tokens']}")
```

### Using Different Storage Providers

```python
# In-memory storage
conversation = Conversation(provider="in-memory")
conversation.add("user", "Hello")

# Mem0 storage
conversation = Conversation(provider="mem0")
conversation.add("user", "Hello")
```

## Conclusion

The `Conversation` class provides a comprehensive set of tools for managing conversations in Python applications. It supports various storage backends, token counting, caching, and multiple export/import formats. The class is designed to be flexible and extensible, making it suitable for a wide range of use cases from simple chat applications to complex conversational AI systems.
