# CLI Multi-Agent Features: 3-Step Quickstart Guide

Run LLM Council and Heavy Swarm directly from the command line for seamless DevOps integration. Execute sophisticated multi-agent workflows without writing Python code.

## Overview

| Feature | Description |
|---------|-------------|
| **LLM Council CLI** | Run collaborative decision-making from terminal |
| **Heavy Swarm CLI** | Execute comprehensive research swarms |
| **DevOps Ready** | Integrate into CI/CD pipelines and scripts |
| **Configurable** | Full parameter control from command line |

---

## Step 1: Install and Verify

Ensure Swarms is installed and verify CLI access:

```bash
# Install swarms
pip install swarms

# Verify CLI is available
swarms --help
```

You should see the Swarms CLI banner and available commands.

---

## Step 2: Set Environment Variables

Configure your API keys:

```bash
# Set your OpenAI API key (or other provider)
export OPENAI_API_KEY="your-openai-api-key"

# Optional: Set workspace directory
export WORKSPACE_DIR="./agent_workspace"
```

Or add to your `.env` file:

```
OPENAI_API_KEY=your-openai-api-key
WORKSPACE_DIR=./agent_workspace
```

---

## Step 3: Run Multi-Agent Commands

### LLM Council

Run a collaborative council of AI agents:

```bash
# Basic usage
swarms llm-council --task "What is the best approach to implement microservices architecture?"

# With verbose output
swarms llm-council --task "Evaluate investment opportunities in AI startups" --verbose
```

### Heavy Swarm

Run comprehensive research and analysis:

```bash
# Basic usage
swarms heavy-swarm --task "Analyze the current state of quantum computing"

# With configuration options
swarms heavy-swarm \
    --task "Research renewable energy market trends" \
    --loops-per-agent 2 \
    --question-agent-model-name gpt-4o-mini \
    --worker-model-name gpt-4o-mini \
    --verbose
```

---

## Complete CLI Reference

### LLM Council Command

```bash
swarms llm-council --task "<your query>" [options]
```

| Option | Description |
|--------|-------------|
| `--task` | **Required.** The query or question for the council |
| `--verbose` | Enable detailed output logging |

**Examples:**

```bash
# Strategic decision
swarms llm-council --task "Should our startup pivot from B2B to B2C?"

# Technical evaluation
swarms llm-council --task "Compare React vs Vue for enterprise applications"

# Business analysis
swarms llm-council --task "What are the risks of expanding to European markets?"
```

---

### Heavy Swarm Command

```bash
swarms heavy-swarm --task "<your task>" [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--task` | - | **Required.** The research task |
| `--loops-per-agent` | 1 | Number of loops per agent |
| `--question-agent-model-name` | gpt-4o-mini | Model for question agent |
| `--worker-model-name` | gpt-4o-mini | Model for worker agents |
| `--random-loops-per-agent` | False | Randomize loops per agent |
| `--verbose` | False | Enable detailed output |

**Examples:**

```bash
# Comprehensive research
swarms heavy-swarm --task "Research the impact of AI on healthcare diagnostics" --verbose

# With custom models
swarms heavy-swarm \
    --task "Analyze cryptocurrency regulation trends globally" \
    --question-agent-model-name gpt-4 \
    --worker-model-name gpt-4 \
    --loops-per-agent 3

# Quick analysis
swarms heavy-swarm --task "Summarize recent advances in battery technology"
```

---

## Other Useful CLI Commands

### Setup Check

Verify your environment is properly configured:

```bash
swarms setup-check --verbose
```

### Run Single Agent

Execute a single agent task:

```bash
swarms agent \
    --name "Research-Agent" \
    --task "Summarize recent AI developments" \
    --model "gpt-4o-mini" \
    --max-loops 1
```

### Auto Swarm

Automatically generate and run a swarm configuration:

```bash
swarms autoswarm --task "Build a content analysis pipeline" --model gpt-4
```

### Show All Commands

Display all available CLI features:

```bash
swarms show-all
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "Command not found" | Ensure `pip install swarms` completed successfully |
| "API key not set" | Export `OPENAI_API_KEY` environment variable |
| "Task cannot be empty" | Always provide `--task` argument |
| Timeout errors | Check network connectivity and API rate limits |

### Debug Mode

Run with verbose output for debugging:

```bash
swarms llm-council --task "Your query" --verbose 2>&1 | tee debug.log
```

---

## Next Steps

- Explore [CLI Reference Documentation](../swarms/cli/cli_reference.md) for all commands
- See [CLI Examples](../swarms/cli/cli_examples.md) for more use cases
- Learn about [LLM Council](./llm_council_quickstart.md) Python API
- Try [Heavy Swarm Documentation](../swarms/structs/heavy_swarm.md) for advanced configuration

