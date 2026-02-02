# Swarms Chat Command Usage

The `swarms chat` command provides an interactive chat agent with optimized defaults for conversational interactions. The agent runs with `max_loops="auto"` for continuous interaction, similar to Claude Code.

## Features

- **Interactive Mode**: Automatically enabled for continuous conversation
- **Auto Loops**: Runs with `max_loops="auto"` for autonomous operation
- **Dynamic Context Window**: Automatically adjusts context window (100,000 tokens)
- **Dynamic Temperature**: Adapts temperature based on conversation
- **Optional Initial Task**: Start with a task or begin with a prompt

## Basic Usage

### Start Chat Without Initial Task
```bash
swarms chat
```

The agent will prompt you for input when started.

### Start Chat With Initial Task
```bash
swarms chat --task "Hello, how can you help me today?"
```

The agent will process the initial task and then continue in interactive mode.

## Advanced Usage

### Custom Agent Name
```bash
swarms chat --name "MyAssistant" --task "Let's discuss Python programming"
```

### Custom System Prompt
```bash
swarms chat --system-prompt "You are an expert Python developer" --task "Help me debug this code"
```

### Full Customization
```bash
swarms chat \
  --name "CodeReviewer" \
  --description "An expert code reviewer specializing in Python" \
  --system-prompt "You are a senior Python developer with expertise in code review and best practices" \
  --task "Review my implementation of a binary search algorithm"
```

## Using Python Module Directly

You can also run the chat command directly with Python:

```bash
python3.12 -m swarms.cli.main chat --task "Hello"
```

## How It Works

1. The chat agent initializes with optimized settings:
   - `interactive=True` - Enables continuous interaction
   - `max_loops="auto"` - Autonomous loop execution
   - `dynamic_context_window=True` - Adaptive context management
   - `dynamic_temperature_enabled=True` - Adaptive response generation
   - `context_length=100000` - Large context window

2. If you provide a `--task`, the agent processes it first
3. After processing, the agent continues to prompt for input
4. You can continue the conversation interactively
5. Exit by typing 'exit', 'quit', or pressing Ctrl+C

## Examples

### Quick Question
```bash
swarms chat --task "What are the best practices for Python async programming?"
```

### Extended Conversation
```bash
swarms chat --name "TutorBot" --system-prompt "You are a patient programming tutor"
```
Then continue with follow-up questions interactively.

### Code Review Session
```bash
swarms chat \
  --name "CodeReviewer" \
  --system-prompt "You are an expert code reviewer. Provide detailed feedback with suggestions." \
  --task "I'll share some code for review"
```

## Tips

- Use `--system-prompt` to customize the agent's behavior and expertise
- Provide a `--task` to start with context before interactive mode
- The agent remembers conversation history within the session
- All standard agent parameters are available for customization

## Troubleshooting

If you encounter the error `auto_chat_agent() got an unexpected keyword argument 'interactive'`, ensure you're using the latest version of Swarms where this has been fixed.

```bash
# Update to latest version
swarms auto-upgrade
```
