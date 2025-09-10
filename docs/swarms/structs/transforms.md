# Message Transforms: Context Management for Large Conversations

The Message Transforms system provides intelligent context size management for AI conversations, automatically handling token limits and message count constraints while preserving the most important contextual information.

## Overview

Message transforms enable agents to handle long conversations that exceed model context windows by intelligently compressing the conversation history. The system uses a "middle-out" compression strategy that preserves system messages, recent messages, and the beginning of conversations while compressing or removing less critical middle content.

## Key Features

- **Automatic Context Management**: Automatically compresses conversations when they approach token or message limits
- **Middle-Out Compression**: Preserves important context (system messages, recent messages, conversation start) while compressing the middle
- **Model-Aware**: Knows context limits for popular models (GPT-4, Claude, etc.)
- **Flexible Configuration**: Highly customizable compression strategies
- **Detailed Logging**: Provides compression statistics and ratios
- **Zero-Configuration Option**: Can work with sensible defaults

## Core Components

### TransformConfig

The configuration class that controls transform behavior:

```python
@dataclass
class TransformConfig:
    enabled: bool = False                    # Enable/disable transforms
    method: str = "middle-out"              # Compression method
    max_tokens: Optional[int] = None        # Token limit override
    max_messages: Optional[int] = None      # Message limit override
    model_name: str = "gpt-4"               # Target model for limit detection
    preserve_system_messages: bool = True   # Always keep system messages
    preserve_recent_messages: int = 2       # Number of recent messages to preserve
```

### TransformResult

Contains the results of message transformation:

```python
@dataclass
class TransformResult:
    messages: List[Dict[str, Any]]          # Transformed message list
    original_token_count: int               # Original token count
    compressed_token_count: int             # New token count after compression
    original_message_count: int             # Original message count
    compressed_message_count: int           # New message count after compression
    compression_ratio: float                # Compression ratio (0.0-1.0)
    was_compressed: bool                    # Whether compression occurred
```

### MessageTransforms

The main transformation engine:

```python
class MessageTransforms:
    def __init__(self, config: TransformConfig):
        """Initialize with configuration."""

    def transform_messages(
        self,
        messages: List[Dict[str, Any]],
        target_model: Optional[str] = None,
    ) -> TransformResult:
        """Transform messages according to configuration."""
```

## Usage Examples

### Basic Agent with Transforms

```python
from swarms import Agent
from swarms.structs.transforms import TransformConfig

# Initialize agent with transforms enabled
agent = Agent(
    agent_name="Trading-Agent",
    agent_description="Financial analysis agent",
    model_name="gpt-4o",
    max_loops=1,
    transforms=TransformConfig(
        enabled=True,
        method="middle-out",
        model_name="gpt-4o",
        preserve_system_messages=True,
        preserve_recent_messages=3,
    ),
)

result = agent.run("Analyze the current market trends...")
```

### Dictionary Configuration

```python
# Alternative dictionary-based configuration
agent = Agent(
    agent_name="Analysis-Agent",
    model_name="claude-3-sonnet",
    transforms={
        "enabled": True,
        "method": "middle-out",
        "model_name": "claude-3-sonnet",
        "preserve_system_messages": True,
        "preserve_recent_messages": 5,
        "max_tokens": 100000,  # Custom token limit
    },
)
```

### Manual Transform Application

```python
from swarms.structs.transforms import MessageTransforms, TransformConfig

# Create transform instance
config = TransformConfig(
    enabled=True,
    model_name="gpt-4",
    preserve_recent_messages=2
)
transforms = MessageTransforms(config)

# Apply to message list
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    # ... many messages ...
    {"role": "user", "content": "What's the weather?"},
]

result = transforms.transform_messages(messages)
print(f"Compressed {result.original_token_count} -> {result.compressed_token_count} tokens")
```

## Compression Strategy

### Middle-Out Algorithm

The middle-out compression strategy works as follows:

1. **Preserve System Messages**: Always keep system messages at the beginning
2. **Preserve Recent Messages**: Keep the most recent N messages
3. **Compress Middle**: Apply compression to messages in the middle of the conversation
4. **Maintain Context Flow**: Ensure the compressed conversation still makes contextual sense

### Token vs Message Limits

The system handles two types of limits:

- **Token Limits**: Based on model's context window (e.g., GPT-4: 8K, Claude: 200K)
- **Message Limits**: Some models limit total message count (e.g., Claude: 1000 messages)

### Smart Model Detection

Built-in knowledge of popular models:

```python
# Supported models include:
"gpt-4": 8192 tokens
"gpt-4-turbo": 128000 tokens
"gpt-4o": 128000 tokens
"claude-3-opus": 200000 tokens
"claude-3-sonnet": 200000 tokens
# ... and many more
```

## Advanced Configuration

### Custom Token Limits

```python
# Override default model limits
config = TransformConfig(
    enabled=True,
    model_name="gpt-4",
    max_tokens=50000,  # Custom limit instead of default 8192
    max_messages=500,  # Custom message limit
)
```

### System Message Preservation

```python
# Fine-tune what gets preserved
config = TransformConfig(
    enabled=True,
    preserve_system_messages=True,  # Keep all system messages
    preserve_recent_messages=5,     # Keep last 5 messages
)
```

## Helper Functions

### Quick Setup

```python
from swarms.structs.transforms import create_default_transforms

# Create with sensible defaults
transforms = create_default_transforms(
    enabled=True,
    model_name="claude-3-sonnet"
)
```

### Direct Application

```python
from swarms.structs.transforms import apply_transforms_to_messages

# Apply transforms to messages directly
result = apply_transforms_to_messages(
    messages=my_messages,
    model_name="gpt-4o"
)
```

## Integration with Agent Memory

Transforms work seamlessly with conversation memory systems:

```python
# Transforms integrate with conversation history
def handle_transforms(
    transforms: MessageTransforms,
    short_memory: Conversation,
    model_name: str = "gpt-4o"
) -> str:
    """Apply transforms to conversation memory."""
    messages = short_memory.return_messages_as_dictionary()
    result = transforms.transform_messages(messages, model_name)

    if result.was_compressed:
        logger.info(f"Compressed conversation: {result.compression_ratio:.2f} ratio")

    return result.messages
```

## Best Practices

### When to Use Transforms

- **Long Conversations**: When conversations exceed model context limits
- **Memory-Intensive Tasks**: Research, analysis, or multi-turn reasoning
- **Production Systems**: Where conversation length is unpredictable
- **Cost Optimization**: Reducing token usage for long conversations

### Configuration Guidelines

- **Start Simple**: Use defaults, then customize based on needs
- **Monitor Compression**: Check logs for compression ratios and effectiveness
- **Preserve Context**: Keep enough recent messages for continuity
- **Test Thoroughly**: Verify compressed conversations maintain quality

### Performance Considerations

- **Token Counting**: Uses efficient tokenization libraries
- **Memory Efficient**: Processes messages in-place when possible
- **Logging Overhead**: Compression stats are logged only when compression occurs
- **Model Compatibility**: Works with any model that has known limits

## Troubleshooting

### Common Issues

**Transforms not activating:**
- Check that `enabled=True` in configuration
- Verify model name matches supported models
- Ensure message count/token count exceeds thresholds

**Poor compression quality:**
- Increase `preserve_recent_messages`
- Ensure system messages are preserved
- Check compression ratios in logs

**Unexpected behavior:**
- Review configuration parameters
- Check model-specific limits
- Examine conversation structure

## API Reference

### TransformConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable/disable transforms |
| `method` | `str` | `"middle-out"` | Compression method |
| `max_tokens` | `Optional[int]` | `None` | Custom token limit |
| `max_messages` | `Optional[int]` | `None` | Custom message limit |
| `model_name` | `str` | `"gpt-4"` | Target model name |
| `preserve_system_messages` | `bool` | `True` | Preserve system messages |
| `preserve_recent_messages` | `int` | `2` | Recent messages to keep |

### TransformResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `messages` | `List[Dict]` | Transformed messages |
| `original_token_count` | `int` | Original token count |
| `compressed_token_count` | `int` | Compressed token count |
| `original_message_count` | `int` | Original message count |
| `compressed_message_count` | `int` | Compressed message count |
| `compression_ratio` | `float` | Compression ratio |
| `was_compressed` | `bool` | Whether compression occurred |

## Examples in Action

### Real-World Use Case: Research Agent

```python
# Research agent that handles long document analysis
research_agent = Agent(
    agent_name="Research-Agent",
    model_name="claude-3-opus",
    transforms=TransformConfig(
        enabled=True,
        model_name="claude-3-opus",
        preserve_recent_messages=5,  # Keep recent context for follow-ups
        max_tokens=150000,  # Leave room for responses
    ),
)

# Agent can now handle very long research conversations
# without hitting context limits
```

### Use Case: Customer Support Bot

```python
# Support bot maintaining conversation history
support_agent = Agent(
    agent_name="Support-Agent",
    model_name="gpt-4o",
    transforms=TransformConfig(
        enabled=True,
        preserve_system_messages=True,
        preserve_recent_messages=10,  # Keep recent conversation
        max_messages=100,  # Reasonable conversation length
    ),
)
```

This comprehensive transforms system ensures that agents can handle conversations of any length while maintaining contextual coherence and optimal performance.
