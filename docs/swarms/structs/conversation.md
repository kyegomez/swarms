# Module/Class Name: Conversation

## Introduction

The `Conversation` class is a powerful and flexible tool for managing conversational data in Python applications. It provides a comprehensive solution for storing, retrieving, and analyzing conversations with support for multiple storage backends, token tracking, and advanced metadata management.

### Key Features

- **Multiple Storage Backends**: Support for various storage solutions:
  - In-memory: Fast, temporary storage for testing and development
  - Supabase: PostgreSQL-based cloud storage with real-time capabilities
  - Redis: High-performance caching and persistence
  - SQLite: Local file-based storage
  - DuckDB: Analytical workloads and columnar storage
  - Pulsar: Event streaming for distributed systems
  - Mem0: Memory-based storage with mem0 integration

- **Token Management**:
  - Built-in token counting with configurable models
  - Automatic token tracking for input/output messages
  - Token usage analytics and reporting
  - Context length management

- **Metadata and Categories**:
  - Support for message metadata
  - Message categorization (input/output)
  - Role-based message tracking
  - Custom message IDs

- **Data Export/Import**:
  - JSON and YAML export formats
  - Automatic saving and loading
  - Conversation history management
  - Batch operations support

- **Advanced Features**:
  - Message search and filtering
  - Conversation analytics
  - Multi-agent support
  - Error handling and fallbacks
  - Type hints and validation

### Use Cases

1. **Chatbot Development**:
   - Store and manage conversation history
   - Track token usage and context length
   - Analyze conversation patterns

2. **Multi-Agent Systems**:
   - Coordinate multiple AI agents
   - Track agent interactions
   - Store agent outputs and metadata

3. **Analytics Applications**:
   - Track conversation metrics
   - Generate usage reports
   - Analyze user interactions

4. **Production Systems**:
   - Persistent storage with various backends
   - Error handling and recovery
   - Scalable conversation management

5. **Development and Testing**:
   - Fast in-memory storage
   - Debugging support
   - Easy export/import of test data

### Best Practices

1. **Storage Selection**:
   - Use in-memory for testing and development
   - Choose Supabase for multi-user cloud applications
   - Use Redis for high-performance requirements
   - Select SQLite for single-user local applications
   - Pick DuckDB for analytical workloads
   - Opt for Pulsar in distributed systems

2. **Token Management**:
   - Enable token counting for production use
   - Set appropriate context lengths
   - Monitor token usage with export_and_count_categories()

3. **Error Handling**:
   - Implement proper fallback mechanisms
   - Use type hints for better code reliability
   - Monitor and log errors appropriately

4. **Data Management**:
   - Use appropriate export formats (JSON/YAML)
   - Implement regular backup strategies
   - Clean up old conversations when needed

5. **Security**:
   - Use environment variables for sensitive credentials
   - Implement proper access controls
   - Validate input data

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
| provider | Literal["mem0", "in-memory"] | "in-memory" | Legacy storage provider |
| backend | Optional[str] | None | Storage backend (takes precedence over provider) |
| tokenizer_model_name | str | "gpt-4.1" | Model name for tokenization |
| conversations_dir | Optional[str] | None | Directory for conversations |
| export_method | str | "json" | Export format ("json" or "yaml") |

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

### Basic Usage with Modern Configuration

```python
from swarms.structs import Conversation

# Create a new conversation with modern configuration
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

### Using Supabase Backend with Environment Variables

```python
import os
from swarms.structs import Conversation

# Using environment variables for secure configuration
os.environ["SUPABASE_URL"] = "https://your-project.supabase.co"
os.environ["SUPABASE_ANON_KEY"] = "your-anon-key"

conversation = Conversation(
    name="supabase_chat",
    backend="supabase",
    system_prompt="You are a helpful assistant",
    time_enabled=True,
    token_count=True,
    message_id_on=True,
    table_name="production_conversations"  # Custom table name
)

# Messages are automatically persisted to Supabase
conversation.add("user", "Hello!", metadata={"client_id": "user123"})
conversation.add("assistant", "Hi there!", metadata={"model": "gpt-4"})

# Search functionality works with backend
results = conversation.search("Hello")
```

### Redis Backend with Advanced Configuration

```python
from swarms.structs import Conversation

# Redis with advanced configuration and persistence
conversation = Conversation(
    name="redis_chat",
    backend="redis",
    redis_host="localhost",
    redis_port=6379,
    redis_password="secure_password",
    use_embedded_redis=False,  # Use external Redis
    persist_redis=True,
    auto_persist=True,
    redis_data_dir="/path/to/redis/data",
    token_count=True
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
```

### SQLite Backend with Custom Path and Export

```python
from swarms.structs import Conversation
import os

# SQLite with custom database path and YAML export
conversation = Conversation(
    name="sqlite_chat",
    backend="sqlite",
    db_path=os.path.expanduser("~/conversations.db"),
    export_method="yaml",
    system_prompt="You are a helpful assistant",
    token_count=True
)

# Add messages and export
conversation.add("user", "Hello SQLite!")
conversation.add("assistant", "Hello from SQLite backend!")

# Export conversation to YAML
conversation.export(force=True)  # force=True overrides autosave setting
```

### Advanced Usage with Multi-Agent Systems

```python
from swarms.structs import Agent, Conversation
from swarms.structs.multi_agent_exec import run_agents_concurrently
import os

# Set up conversation with DuckDB backend for analytics
conversation = Conversation(
    name="multi_agent_analytics",
    backend="duckdb",
    db_path="analytics.duckdb",
    system_prompt="Multi-agent analytics session",
    time_enabled=True,
    token_count=True,
    message_id_on=True
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
```

### Error Handling and Fallbacks with Type Hints

```python
from typing import Optional, Dict, Any
from swarms.structs import Conversation

def initialize_conversation(
    name: str,
    backend: str,
    config: Dict[str, Any]
) -> Optional[Conversation]:
    """Initialize conversation with fallback handling."""
    try:
        conversation = Conversation(
            name=name,
            backend=backend,
            **config
        )
        print(f"✅ {backend} backend initialized successfully")
        return conversation
    except ImportError as e:
        print(f"❌ {backend} not available: {e}")
        # Fallback to in-memory with same configuration
        fallback_config = {
            k: v for k, v in config.items() 
            if k not in ['supabase_url', 'supabase_key', 'redis_host']
        }
        conversation = Conversation(
            name=name,
            backend="in-memory",
            **fallback_config
        )
        print("💡 Falling back to in-memory storage")
        return conversation
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

# Usage
config = {
    "system_prompt": "You are a helpful assistant",
    "time_enabled": True,
    "token_count": True,
    "supabase_url": "https://your-project.supabase.co",
    "supabase_key": "your-key"
}

conversation = initialize_conversation(
    name="fallback_test",
    backend="supabase",
    config=config
)

if conversation:
    conversation.add("user", "Hello!")
```

### Loading and Managing Conversations with Modern Features

```python
from swarms.structs import Conversation
from typing import List, Dict
import os

def manage_conversations(base_dir: str) -> List[Dict[str, str]]:
    """Manage conversations with modern features."""
    
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

### Backend Comparison and Selection Guide

```python
# In-memory: Fast, no persistence, good for testing
conv_memory = Conversation(
    backend="in-memory",
    token_count=True,
    message_id_on=True
)

# SQLite: Local file-based persistence, good for single-user apps
conv_sqlite = Conversation(
    backend="sqlite",
    db_path="conversations.db",
    token_count=True,
    export_method="json"
)

# Redis: High performance, good for real-time applications
conv_redis = Conversation(
    backend="redis",
    redis_host="localhost",
    persist_redis=True,
    token_count=True
)

# Supabase: Cloud PostgreSQL, good for multi-user applications
conv_supabase = Conversation(
    backend="supabase", 
    supabase_url=os.getenv("SUPABASE_URL"),
    supabase_key=os.getenv("SUPABASE_ANON_KEY"),
    token_count=True
)

# DuckDB: Analytical workloads, good for data analysis
conv_duckdb = Conversation(
    backend="duckdb",
    db_path="analytics.duckdb",
    token_count=True
)

# Pulsar: Event streaming, good for distributed systems
conv_pulsar = Conversation(
    backend="pulsar",
    token_count=True
)
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
