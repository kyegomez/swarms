# CLI Agent Guide: Create Agents from Command Line

Create, configure, and run AI agents directly from your terminal without writing Python code.

## Basic Agent Creation

### Step 1: Define Your Agent

Create an agent with required parameters:

```bash
swarms agent \
    --name "Research-Agent" \
    --description "An AI agent that researches topics and provides summaries" \
    --system-prompt "You are an expert researcher. Provide comprehensive, well-structured summaries with key insights." \
    --task "Research the current state of quantum computing and its applications"
```

### Step 2: Customize Model Settings

Add model configuration options:

```bash
swarms agent \
    --name "Code-Reviewer" \
    --description "Expert code review assistant" \
    --system-prompt "You are a senior software engineer. Review code for best practices, bugs, and improvements." \
    --task "Review this Python function for efficiency: def fib(n): return fib(n-1) + fib(n-2) if n > 1 else n" \
    --model-name "gpt-4o-mini" \
    --temperature 0.1 \
    --max-loops 3
```

### Step 3: Enable Advanced Features

Add streaming, dashboard, and autosave:

```bash
swarms agent \
    --name "Analysis-Agent" \
    --description "Data analysis specialist" \
    --system-prompt "You are a data analyst. Provide detailed statistical analysis and insights." \
    --task "Analyze market trends for electric vehicles in 2024" \
    --model-name "gpt-4" \
    --streaming-on \
    --verbose \
    --autosave \
    --saved-state-path "./agent_states/analysis_agent.json"
```

---

## Complete Parameter Reference

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--name` | Agent name | `"Research-Agent"` |
| `--description` | Agent description | `"AI research assistant"` |
| `--system-prompt` | Agent's system instructions | `"You are an expert..."` |
| `--task` | Task for the agent | `"Analyze this data"` |

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-name` | `"gpt-4"` | LLM model to use |
| `--temperature` | `None` | Creativity (0.0-2.0) |
| `--max-loops` | `None` | Maximum execution loops |
| `--context-length` | `None` | Context window size |

### Behavior Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--auto-generate-prompt` | `False` | Auto-generate prompts |
| `--dynamic-temperature-enabled` | `False` | Dynamic temperature adjustment |
| `--dynamic-context-window` | `False` | Dynamic context window |
| `--streaming-on` | `False` | Enable streaming output |
| `--verbose` | `False` | Verbose mode |

### State Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--autosave` | `False` | Enable autosave |
| `--saved-state-path` | `None` | Path to save state |
| `--dashboard` | `False` | Enable dashboard |
| `--return-step-meta` | `False` | Return step metadata |

### Integration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mcp-url` | `None` | MCP server URL |
| `--user-name` | `None` | Username for agent |
| `--output-type` | `None` | Output format (str, json) |
| `--retry-attempts` | `None` | Retry attempts on failure |

---

## Use Case Examples

### Financial Analyst Agent

```bash
swarms agent \
    --name "Financial-Analyst" \
    --description "Expert financial analysis and market insights" \
    --system-prompt "You are a CFA-certified financial analyst. Provide detailed market analysis with data-driven insights. Include risk assessments and recommendations." \
    --task "Analyze Apple (AAPL) stock performance and provide investment outlook for Q4 2024" \
    --model-name "gpt-4" \
    --temperature 0.2 \
    --max-loops 5 \
    --verbose
```

### Code Generation Agent

```bash
swarms agent \
    --name "Code-Generator" \
    --description "Expert Python developer and code generator" \
    --system-prompt "You are an expert Python developer. Write clean, efficient, well-documented code following PEP 8 guidelines. Include type hints and docstrings." \
    --task "Create a Python class for managing a task queue with priority scheduling" \
    --model-name "gpt-4" \
    --temperature 0.1 \
    --streaming-on
```

### Creative Writing Agent

```bash
swarms agent \
    --name "Creative-Writer" \
    --description "Professional content writer and storyteller" \
    --system-prompt "You are a professional writer with expertise in engaging content. Write compelling, creative content with strong narrative flow." \
    --task "Write a short story about a scientist who discovers time travel" \
    --model-name "gpt-4" \
    --temperature 0.8 \
    --max-loops 2
```

### Research Summarizer Agent

```bash
swarms agent \
    --name "Research-Summarizer" \
    --description "Academic research summarization specialist" \
    --system-prompt "You are an academic researcher. Summarize research topics with key findings, methodologies, and implications. Cite sources when available." \
    --task "Summarize recent advances in CRISPR gene editing technology" \
    --model-name "gpt-4o-mini" \
    --temperature 0.3 \
    --verbose \
    --autosave
```

---

## Scripting Examples

### Bash Script with Multiple Agents

```bash
#!/bin/bash
# run_agents.sh

# Research phase
swarms agent \
    --name "Researcher" \
    --description "Research specialist" \
    --system-prompt "You are a researcher. Gather comprehensive information on topics." \
    --task "Research the impact of AI on healthcare" \
    --model-name "gpt-4o-mini" \
    --output-type "json" > research_output.json

# Analysis phase
swarms agent \
    --name "Analyst" \
    --description "Data analyst" \
    --system-prompt "You are an analyst. Analyze data and provide insights." \
    --task "Analyze the research findings from: $(cat research_output.json)" \
    --model-name "gpt-4o-mini" \
    --output-type "json" > analysis_output.json

echo "Pipeline complete!"
```

### Loop Through Tasks

```bash
#!/bin/bash
# batch_analysis.sh

TOPICS=("renewable energy" "electric vehicles" "smart cities" "AI ethics")

for topic in "${TOPICS[@]}"; do
    echo "Analyzing: $topic"
    swarms agent \
        --name "Topic-Analyst" \
        --description "Topic analysis specialist" \
        --system-prompt "You are an expert analyst. Provide concise analysis of topics." \
        --task "Analyze current trends in: $topic" \
        --model-name "gpt-4o-mini" \
        >> "analysis_results.txt"
    echo "---" >> "analysis_results.txt"
done
```

---

## Tips and Best Practices

!!! tip "System Prompt Tips"
    - Be specific about the agent's role and expertise
    - Include output format preferences
    - Specify any constraints or guidelines

!!! tip "Temperature Settings"
    - Use **0.1-0.3** for factual/analytical tasks
    - Use **0.5-0.7** for balanced responses
    - Use **0.8-1.0** for creative tasks

!!! tip "Performance Optimization"
    - Use `gpt-4o-mini` for simpler tasks (faster, cheaper)
    - Use `gpt-4` for complex reasoning tasks
    - Set appropriate `--max-loops` to control execution time

!!! warning "Common Issues"
    - Ensure API key is set: `export OPENAI_API_KEY="..."`
    - Wrap multi-word arguments in quotes
    - Use `--verbose` to debug issues

---

## Next Steps

- [CLI YAML Configuration](./cli_yaml_guide.md) - Run agents from YAML files
- [CLI Multi-Agent Guide](../examples/cli_multi_agent_quickstart.md) - LLM Council and Heavy Swarm
- [CLI Reference](./cli_reference.md) - Complete command documentation

