# Module/Class Name: Conversation

## Introduction

The `Conversation` class is a powerful tool for managing and structuring conversation data in a Python program. It enables you to create, manipulate, and analyze conversations easily with support for multiple storage backends including persistent databases. This documentation provides a comprehensive understanding of the `Conversation` class, its attributes, methods, and how to effectively use it with different storage backends.

## Table of Contents

1. [Class Definition](#1-class-definition)
2. [Initialization Parameters](#2-initialization-parameters)
3. [Backend Configuration](#3-backend-configuration)
4. [Methods](#4-methods)
5. [Examples](#5-examples)

## 1. Class Definition

### Overview

The `Conversation` class is designed to manage conversations by keeping track of messages and their attributes. It offers methods for adding, deleting, updating, querying, and displaying messages within the conversation. Additionally, it supports exporting and importing conversations, searching for specific keywords, and more.

**New in this version**: The class now supports multiple storage backends for persistent conversation storage:

- **"in-memory"**: Default memory-based storage (no persistence)
- **"mem0"**: Memory-based storage with mem0 integration (requires: `pip install mem0ai`)
- **"supabase"**: PostgreSQL-based storage using Supabase (requires: `pip install supabase`)
- **"redis"**: Redis-based storage (requires: `pip install redis`)
- **"sqlite"**: SQLite-based storage (built-in to Python)
- **"duckdb"**: DuckDB-based storage (requires: `pip install duckdb`)
- **"pulsar"**: Apache Pulsar messaging backend (requires: `pip install pulsar-client`)

All backends use **lazy loading** - database dependencies are only imported when the specific backend is instantiated. Each backend provides helpful error messages if required packages are not installed.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| id | str | Unique identifier for the conversation |
| name | str | Name of the conversation |
| system_prompt | Optional[str] | System prompt for the conversation |
| time_enabled | bool | Flag to enable time tracking for messages |
| autosave | bool | Flag to enable automatic saving |
| save_enabled | bool | Flag to control if saving is enabled |
| save_filepath | str | File path for saving conversation history |
| load_filepath | str | File path for loading conversation history |
| conversation_history | list | List storing conversation messages |
| tokenizer | Callable | Tokenizer for counting tokens |
| context_length | int | Maximum tokens allowed in conversation |
| rules | str | Rules for the conversation |
| custom_rules_prompt | str | Custom prompt for rules |
| user | str | User identifier for messages |
| save_as_yaml | bool | Flag to save as YAML |
| save_as_json_bool | bool | Flag to save as JSON |
| token_count | bool | Flag to enable token counting |
| message_id_on | bool | Flag to enable message IDs |
| backend | str | Storage backend type |
| backend_instance | Any | The actual backend instance |
| conversations_dir | str | Directory to store conversations |

## 2. Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| id | str | generated | Unique conversation ID |
| name | str | None | Name of the conversation |
| system_prompt | Optional[str] | None | System prompt for the conversation |
| time_enabled | bool | False | Enable time tracking |
| autosave | bool | False | Enable automatic saving |
| save_enabled | bool | False | Control if saving is enabled |
| save_filepath | str | None | File path for saving |
| load_filepath | str | None | File path for loading |
| tokenizer | Callable | None | Tokenizer for counting tokens |
| context_length | int | 8192 | Maximum tokens allowed |
| rules | str | None | Conversation rules |
| custom_rules_prompt | str | None | Custom rules prompt |
| user | str | "User:" | User identifier |
| save_as_yaml | bool | False | Save as YAML |
| save_as_json_bool | bool | False | Save as JSON |
| token_count | bool | True | Enable token counting |
| message_id_on | bool | False | Enable message IDs |
| provider | Literal["mem0", "in-memory"] | "in-memory" | Legacy storage provider |
| backend | Optional[str] | None | Storage backend (takes precedence over provider) |
| conversations_dir | Optional[str] | None | Directory for conversations |

## 3. Backend Configuration

### Backend-Specific Parameters

#### Supabase Backend
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| supabase_url | Optional[str] | None | Supabase project URL |
| supabase_key | Optional[str] | None | Supabase API key |
| table_name | str | "conversations" | Database table name |

Environment variables: `SUPABASE_URL`, `SUPABASE_ANON_KEY`

#### Redis Backend
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| redis_host | str | "localhost" | Redis server host |
| redis_port | int | 6379 | Redis server port |
| redis_db | int | 0 | Redis database number |
| redis_password | Optional[str] | None | Redis password |
| use_embedded_redis | bool | True | Use embedded Redis |
| persist_redis | bool | True | Enable Redis persistence |
| auto_persist | bool | True | Auto-persist data |
| redis_data_dir | Optional[str] | None | Redis data directory |

#### SQLite/DuckDB Backend
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| db_path | Optional[str] | None | Database file path |

#### Pulsar Backend
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| pulsar_url | str | "pulsar://localhost:6650" | Pulsar server URL |
| topic | str | f"conversation-{id}" | Pulsar topic name |

### Backend Selection

The `backend` parameter takes precedence over the legacy `provider` parameter:

```python
# Legacy way (still supported)
conversation = Conversation(provider="in-memory")

# New way (recommended)
conversation = Conversation(backend="supabase")
conversation = Conversation(backend="redis")
conversation = Conversation(backend="sqlite")
```

## 4. Methods

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

## 5. Examples

### Basic Usage

```python
from swarms.structs import Conversation

# Create a new conversation with in-memory storage
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

# Save conversation (in-memory only saves to file)
conversation.save_as_json("my_chat.json")
```

### Using Supabase Backend

```python
import os
from swarms.structs import Conversation

# Using environment variables
os.environ["SUPABASE_URL"] = "https://your-project.supabase.co"
os.environ["SUPABASE_ANON_KEY"] = "your-anon-key"

conversation = Conversation(
    name="supabase_chat",
    backend="supabase",
    system_prompt="You are a helpful assistant",
    time_enabled=True
)

# Or using explicit parameters
conversation = Conversation(
    name="supabase_chat",
    backend="supabase",
    supabase_url="https://your-project.supabase.co",
    supabase_key="your-anon-key",
    system_prompt="You are a helpful assistant",
    time_enabled=True
)

# Add messages (automatically stored in Supabase)
conversation.add("user", "Hello!")
conversation.add("assistant", "Hi there!")

# All operations work transparently with the backend
conversation.display_conversation()
results = conversation.search("Hello")
```

### Using Redis Backend

```python
from swarms.structs import Conversation

# Using Redis with default settings
conversation = Conversation(
    name="redis_chat",
    backend="redis",
    system_prompt="You are a helpful assistant"
)

# Using Redis with custom configuration
conversation = Conversation(
    name="redis_chat",
    backend="redis",
    redis_host="localhost",
    redis_port=6379,
    redis_db=0,
    redis_password="mypassword",
    system_prompt="You are a helpful assistant"
)

conversation.add("user", "Hello Redis!")
conversation.add("assistant", "Hello from Redis backend!")
```

### Using SQLite Backend

```python
from swarms.structs import Conversation

# SQLite with default database file
conversation = Conversation(
    name="sqlite_chat",
    backend="sqlite",
    system_prompt="You are a helpful assistant"
)

# SQLite with custom database path
conversation = Conversation(
    name="sqlite_chat",
    backend="sqlite",
    db_path="/path/to/my/conversations.db",
    system_prompt="You are a helpful assistant"
)

conversation.add("user", "Hello SQLite!")
conversation.add("assistant", "Hello from SQLite backend!")
```

### Advanced Usage with Multi-Agent Systems

```python
import os
from swarms.structs import Agent, Conversation
from swarms.structs.multi_agent_exec import run_agents_concurrently

# Set up Supabase backend for persistent storage
conversation = Conversation(
    name="multi_agent_research",
    backend="supabase",
    supabase_url=os.getenv("SUPABASE_URL"),
    supabase_key=os.getenv("SUPABASE_ANON_KEY"),
    system_prompt="Multi-agent collaboration session",
    time_enabled=True
)

# Create specialized agents
data_analyst = Agent(
    agent_name="DataAnalyst",
    system_prompt="You are a senior data analyst...",
    model_name="gpt-4o-mini",
    max_loops=1
)

researcher = Agent(
    agent_name="ResearchSpecialist", 
    system_prompt="You are a research specialist...",
    model_name="gpt-4o-mini",
    max_loops=1
)

# Run agents and store results in persistent backend
task = "Analyze the current state of AI in healthcare"
results = run_agents_concurrently(agents=[data_analyst, researcher], task=task)

# Store results in conversation (automatically persisted)
for result, agent in zip(results, [data_analyst, researcher]):
    conversation.add(content=result, role=agent.agent_name)

# Conversation is automatically saved to Supabase
print(f"Conversation stored with {len(conversation.to_dict())} messages")
```

### Error Handling and Fallbacks

```python
from swarms.structs import Conversation

try:
    # Attempt to use Supabase backend
    conversation = Conversation(
        name="fallback_test",
        backend="supabase",
        supabase_url="https://your-project.supabase.co",
        supabase_key="your-key"
    )
    print("✅ Supabase backend initialized successfully")
except ImportError as e:
    print(f"❌ Supabase not available: {e}")
    # Automatic fallback to in-memory storage
    conversation = Conversation(
        name="fallback_test",
        backend="in-memory"
    )
    print("💡 Falling back to in-memory storage")

# Usage remains the same regardless of backend
conversation.add("user", "Hello!")
conversation.add("assistant", "Hi there!")
```

### Loading and Managing Conversations

```python
from swarms.structs import Conversation

# List all saved conversations
conversations = Conversation.list_conversations()
for conv in conversations:
    print(f"ID: {conv['id']}, Name: {conv['name']}, Created: {conv['created_at']}")

# Load a specific conversation
conversation = Conversation.load_conversation("my_conversation_name")

# Load conversation from specific file
conversation = Conversation.load_conversation(
    "my_chat",
    load_filepath="/path/to/conversation.json"
)
```

### Backend Comparison

```python
# In-memory: Fast, no persistence
conv_memory = Conversation(backend="in-memory")

# SQLite: Local file-based persistence
conv_sqlite = Conversation(backend="sqlite", db_path="conversations.db")

# Redis: Distributed caching, high performance
conv_redis = Conversation(backend="redis", redis_host="localhost")

# Supabase: Cloud PostgreSQL, real-time features
conv_supabase = Conversation(
    backend="supabase", 
    supabase_url="https://project.supabase.co",
    supabase_key="your-key"
)

# DuckDB: Analytical workloads, columnar storage
conv_duckdb = Conversation(backend="duckdb", db_path="analytics.duckdb")
```

## Error Handling

The conversation class provides graceful error handling:

- **Missing Dependencies**: Clear error messages with installation instructions
- **Backend Failures**: Automatic fallback to in-memory storage
- **Network Issues**: Retry logic and connection management
- **Data Corruption**: Validation and recovery mechanisms

Example error message:
```
Backend 'supabase' dependencies not available. Install with: pip install supabase
```

## Migration Guide

### From Provider to Backend

```python
# Old way
conversation = Conversation(provider="in-memory")

# New way (recommended)
conversation = Conversation(backend="in-memory")

# Both work, but backend takes precedence
conversation = Conversation(
    provider="in-memory",  # Ignored
    backend="supabase"     # Used
)
```

## Conclusion

The `Conversation` class provides a comprehensive set of tools for managing conversations in Python applications with full backend flexibility. It supports various storage backends, lazy loading, token counting, caching, and multiple export/import formats. The class is designed to be flexible and extensible, making it suitable for a wide range of use cases from simple chat applications to complex conversational AI systems with persistent storage requirements.

Choose the appropriate backend based on your needs:
- **in-memory**: Development and testing
- **sqlite**: Local applications and small-scale deployments  
- **redis**: Distributed applications requiring high performance
- **supabase**: Cloud applications with real-time requirements
- **duckdb**: Analytics and data science workloads
- **pulsar**: Event-driven architectures and streaming applications
