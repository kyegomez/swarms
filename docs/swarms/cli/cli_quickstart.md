# CLI Quickstart: Getting Started in 3 Steps

Get up and running with the Swarms CLI in minutes. This guide covers installation, setup verification, and running your first commands.

## Step 1: Install Swarms

Install the Swarms package which includes the CLI:

```bash
pip install swarms
```

Verify installation:

```bash
swarms --help
```

You should see the Swarms CLI banner with available commands.

---

## Step 2: Configure Environment

Set up your API keys and workspace:

```bash
# Set your OpenAI API key (or other provider)
export OPENAI_API_KEY="your-openai-api-key"

# Optional: Set workspace directory
export WORKSPACE_DIR="./agent_workspace"
```

Or create a `.env` file in your project directory:

```
OPENAI_API_KEY=your-openai-api-key
WORKSPACE_DIR=./agent_workspace
```

Verify your setup:

```bash
swarms setup-check --verbose
```

Expected output:

```
🔍 Running Swarms Environment Setup Check

┌─────────────────────────────────────────────────────────────────────────────┐
│ Environment Check Results                                                  │
├─────────┬─────────────────────────┬─────────────────────────────────────────┤
│ Status  │ Check                   │ Details                                │
├─────────┼─────────────────────────┼─────────────────────────────────────────┤
│ ✓       │ Python Version          │ Python 3.11.5                          │
│ ✓       │ Swarms Version          │ Current version: 8.7.0                 │
│ ✓       │ API Keys                │ API keys found: OPENAI_API_KEY         │
│ ✓       │ Dependencies            │ All required dependencies available     │
└─────────┴─────────────────────────┴─────────────────────────────────────────┘
```

---

## Step 3: Run Your First Command

Try these commands to verify everything works:

### Start an Interactive Chat

```bash
swarms chat
```

### Create a Simple Agent

```bash
swarms agent \
    --name "Assistant" \
    --description "A helpful AI assistant" \
    --system-prompt "You are a helpful assistant that provides clear, concise answers." \
    --task "What are the benefits of renewable energy?" \
    --model-name "gpt-4o-mini"
```

### Run LLM Council

```bash
swarms llm-council --task "What are the best practices for code review?"
```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `swarms --help` | Show all available commands |
| `swarms chat` | Start an interactive chat agent |
| `swarms setup-check` | Verify environment setup |
| `swarms onboarding` | Interactive setup wizard |
| `swarms agent` | Create and run a custom agent |
| `swarms llm-council` | Run collaborative LLM council |
| `swarms heavy-swarm` | Run comprehensive analysis swarm |

---

## Next Steps

- [CLI Agent Guide](./cli_agent_guide.md) - Create custom agents from CLI
- [CLI Multi-Agent Guide](../examples/cli_multi_agent_quickstart.md) - Run LLM Council and Heavy Swarm
- [CLI Reference](./cli_reference.md) - Complete command documentation

