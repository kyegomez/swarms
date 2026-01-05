# Swarms CLI Reference

The Swarms CLI is a comprehensive command-line interface for managing and executing Swarms agents and multi-agent architectures. This reference documents all available commands, arguments, and features.

## Table of Contents

- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Commands Reference](#commands-reference)
- [Global Arguments](#global-arguments)
- [Command-Specific Arguments](#command-specific-arguments)
  - [run-agents Command](#run-agents-command)
  - [load-markdown Command](#load-markdown-command)
  - [agent Command](#agent-command)
  - [autoswarm Command](#autoswarm-command)
  - [setup-check Command](#setup-check-command)
  - [llm-council Command](#llm-council-command)
  - [heavy-swarm Command](#heavy-swarm-command)
  - [marketplace Command](#marketplace-command)
  - [features Command](#features-command)
- [Error Handling](#error-handling)
- [Examples](#examples)
- [Configuration](#configuration)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Integration](#integration)
- [Performance Considerations](#performance-considerations)
- [Security](#security)
- [Command Quick Reference](#command-quick-reference)
- [Support](#support)


## Installation

The CLI is included with the Swarms package installation:

```bash
pip install swarms
```

## Basic Usage

```bash
swarms <command> [options]
```

## Commands Reference

### Core Commands

| Command | Description | Required Arguments |
|---------|-------------|-------------------|
| `onboarding` | Start interactive onboarding process | None |
| `help` | Display help message | None |
| `features` | Display all available features and actions in a comprehensive table | None |
| `get-api-key` | Open API key portal in browser | None |
| `check-login` | Verify login status and initialize cache | None |
| `run-agents` | Execute agents from YAML configuration | `--yaml-file` |
| `load-markdown` | Load agents from markdown files | `--markdown-path` |
| `agent` | Create and run custom agent | `--name`, `--description`, `--system-prompt`, `--task` |
| `auto-upgrade` | Update Swarms to latest version | None |
| `book-call` | Schedule strategy session | None |
| `autoswarm` | Generate and execute autonomous swarm | `--task`, `--model` |
| `setup-check` | Run comprehensive environment setup check | None |
| `llm-council` | Run LLM Council with multiple agents collaborating on a task | `--task` |
| `heavy-swarm` | Run HeavySwarm with specialized agents for complex task analysis | `--task` |
| `marketplace` | Search, browse, and install agents from Swarms Marketplace | Subcommand |

## Global Arguments

All commands support these global options:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--verbose` | `bool` | `False` | Enable verbose output |
| `--help`, `-h` | `bool` | `False` | Show help message |

## Command-Specific Arguments

### `run-agents` Command

Execute agents from YAML configuration files.

```bash
python -m swarms.cli.main run-agents [options]
```

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--yaml-file` | `str` | `"agents.yaml"` | No | Path to YAML configuration file |

**Example:**
```bash
swarms run-agents --yaml-file my_agents.yaml
```

### `load-markdown` Command

Load agents from markdown files with YAML frontmatter.

```bash
python -m swarms.cli.main load-markdown [options]
```

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--markdown-path` | `str` | `None` | **Yes** | Path to markdown file or directory |
| `--concurrent` | `bool` | `True` | No | Enable concurrent processing for multiple files |

**Example:**
```bash
swarms load-markdown --markdown-path ./agents/ --concurrent
```

### `agent` Command

Create and run a custom agent with specified parameters.

```bash
python -m swarms.cli.main agent [options]
```

#### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--name` | `str` | Name of the custom agent |
| `--description` | `str` | Description of the custom agent |
| `--system-prompt` | `str` | System prompt for the custom agent |
| `--task` | `str` | Task for the custom agent to execute |

#### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-name` | `str` | `"gpt-4"` | Model name for the custom agent |
| `--temperature` | `float` | `None` | Temperature setting (0.0-2.0) |
| `--max-loops` | `int` | `None` | Maximum number of loops for the agent |
| `--auto-generate-prompt` | `bool` | `False` | Enable auto-generation of prompts |
| `--dynamic-temperature-enabled` | `bool` | `False` | Enable dynamic temperature adjustment |
| `--dynamic-context-window` | `bool` | `False` | Enable dynamic context window |
| `--output-type` | `str` | `None` | Output type (e.g., 'str', 'json') |
| `--verbose` | `bool` | `False` | Enable verbose mode for the agent |
| `--streaming-on` | `bool` | `False` | Enable streaming mode for the agent |
| `--context-length` | `int` | `None` | Context length for the agent |
| `--retry-attempts` | `int` | `None` | Number of retry attempts for the agent |
| `--return-step-meta` | `bool` | `False` | Return step metadata from the agent |
| `--dashboard` | `bool` | `False` | Enable dashboard for the agent |
| `--autosave` | `bool` | `False` | Enable autosave for the agent |
| `--saved-state-path` | `str` | `None` | Path for saving agent state |
| `--user-name` | `str` | `None` | Username for the agent |
| `--mcp-url` | `str` | `None` | MCP URL for the agent |

**Example:**
```bash
swarms agent \
  --name "Trading Agent" \
  --description "Advanced trading agent for market analysis" \
  --system-prompt "You are an expert trader..." \
  --task "Analyze market trends for AAPL" \
  --model-name "gpt-4" \
  --temperature 0.1 \
  --max-loops 5
```

### `autoswarm` Command

Generate and execute an autonomous swarm configuration.

```bash
swarms autoswarm [options]
```

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--task` | `str` | `None` | **Yes** | Task description for the swarm |
| `--model` | `str` | `None` | **Yes** | Model name to use for the swarm |

**Example:**

```bash
swarms autoswarm --task "analyze this data" --model "gpt-4"
```

### `setup-check` Command

Run a comprehensive environment setup check to verify your Swarms installation and configuration.

```bash
swarms setup-check [--verbose]
```

**Arguments:**
- `--verbose`: Enable detailed debug output showing version detection methods

This command performs the following checks:
- **Python Version**: Verifies Python 3.10+ compatibility
- **Swarms Version**: Checks current version and compares with latest available
- **API Keys**: Verifies presence of common API keys in environment variables
- **Dependencies**: Ensures required packages are available
- **Environment File**: Checks for .env file existence and content
- **Workspace Directory**: Verifies WORKSPACE_DIR environment variable

**Examples:**
```bash
# Basic setup check
swarms setup-check

# Verbose setup check with debug information
swarms setup-check --verbose
```

**Expected Output:**
```
🔍 Running Swarms Environment Setup Check

┌─────────────────────────────────────────────────────────────────────────────┐
│ Environment Check Results                                                  │
├─────────┬─────────────────────────┬─────────────────────────────────────────┤
│ Status  │ Check                   │ Details                                │
├─────────┼─────────────────────────┼─────────────────────────────────────────┤
│ ✓       │ Python Version          │ Python 3.11.5                          │
│ ✓       │ Swarms Version          │ Current version: 8.1.1                 │
│ ✓       │ API Keys                │ API keys found: OPENAI_API_KEY         │
│ ✓       │ Dependencies            │ All required dependencies available     │
│ ✓       │ Environment File        │ .env file exists with 1 API key(s)     │
│ ✓       │ Workspace Directory     │ WORKSPACE_DIR is set to: /path/to/ws   │
└─────────┴─────────────────────────┴─────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ Setup Check Complete                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ 🎉 All checks passed! Your environment is ready for Swarms.               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### `llm-council` Command

Run the LLM Council with multiple specialized agents that collaborate, evaluate, and synthesize responses.

The LLM Council follows a structured workflow:
1. **Independent Responses**: Each council member (GPT-5.1, Gemini 3 Pro, Claude Sonnet 4.5, Grok-4) independently responds to the query
2. **Peer Review**: All members review and rank each other's anonymized responses
3. **Synthesis**: A Chairman agent synthesizes all responses and rankings into a final comprehensive answer

```bash
swarms llm-council [options]
```

#### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--task` | `str` | The query or question for the LLM Council to process |

#### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--verbose` | `bool` | `True` | Enable verbose output showing progress and intermediate results |

**Example:**
```bash
# Basic usage
swarms llm-council --task "What are the best energy ETFs right now?"

# With verbose output
swarms llm-council --task "What is the best approach to solve this problem?" --verbose
```

**How It Works:**

The LLM Council creates a collaborative environment where:
- **Default Council Members**: GPT-5.1 (analytical), Gemini 3 Pro (concise), Claude Sonnet 4.5 (balanced), Grok-4 (creative)
- **Anonymized Evaluation**: Responses are anonymized before evaluation to ensure honest ranking
- **Cross-Model Evaluation**: Each model evaluates all responses, often selecting other models' responses as superior
- **Final Synthesis**: The Chairman (GPT-5.1 by default) synthesizes the best elements from all responses

**Use Cases:**
- Complex problem-solving requiring multiple perspectives
- Research questions needing comprehensive analysis
- Decision-making scenarios requiring thorough evaluation
- Content generation with quality assurance

### `heavy-swarm` Command

Run HeavySwarm with specialized agents for complex task analysis and decomposition.

HeavySwarm follows a structured workflow:
1. **Task Decomposition**: Breaks down tasks into specialized questions
2. **Parallel Execution**: Executes specialized agents in parallel
3. **Result Synthesis**: Integrates and synthesizes results
4. **Comprehensive Reporting**: Generates detailed final reports
5. **Iterative Refinement**: Optional multi-loop execution for iterative improvement

```bash
swarms heavy-swarm [options]
```

#### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--task` | `str` | The task for HeavySwarm to analyze and process |

#### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--loops-per-agent` | `int` | `1` | Number of execution loops each agent should perform |
| `--question-agent-model-name` | `str` | `"gpt-4o-mini"` | Model name for the question generation agent |
| `--worker-model-name` | `str` | `"gpt-4o-mini"` | Model name for specialized worker agents |
| `--random-loops-per-agent` | `bool` | `False` | Enable random number of loops per agent (1-10 range) |
| `--verbose` | `bool` | `False` | Enable verbose output showing detailed progress |

**Example:**
```bash
# Basic usage
swarms heavy-swarm --task "Analyze the current market trends for renewable energy"

# With custom configuration
swarms heavy-swarm \
  --task "Research the best investment strategies for 2024" \
  --loops-per-agent 3 \
  --question-agent-model-name "gpt-4" \
  --worker-model-name "gpt-4" \
  --random-loops-per-agent \
  --verbose
```

**Specialized Agent Roles:**

HeavySwarm includes specialized agents for different aspects of analysis:
- **Research Agent**: Fast, trustworthy, and reproducible research
- **Analysis Agent**: Statistical analysis and validated insights
- **Writing Agent**: Clear, structured documentation
- **Question Agent**: Task decomposition and question generation

**Use Cases:**
- Complex research tasks requiring multiple perspectives
- Market analysis and financial research
- Technical analysis and evaluation
- Comprehensive report generation
- Multi-faceted problem solving

### `marketplace` Command

Search, browse, and install agents from the [Swarms Marketplace](https://swarms.world).

```bash
swarms marketplace <action> [options]
```

#### Actions

| Action | Description |
|--------|-------------|
| `search` | Search for agents by keyword |
| `list` | List available agents |
| `info <id>` | View agent details |
| `install <id>` | Install agent locally |

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--query` | `str` | None | Search keyword for agent names/descriptions |
| `--category` | `str` | None | Filter by category |
| `--free-only` | `bool` | `False` | Show only free agents |
| `--limit` | `int` | `20` | Maximum results (1-100) |
| `--output-dir` | `str` | `.` | Directory for installed agents |

**Examples:**
```bash
# Show marketplace help
swarms marketplace

# List available agents
swarms marketplace list --limit 10

# Search for agents
swarms marketplace search --query "trading" --category "finance"

# View agent details
swarms marketplace info a2e3d0d3-9b6a-40a3-9904-000f2e1d03e3

# Install agent to directory
swarms marketplace install a2e3d0d3-9b6a-40a3-9904-000f2e1d03e3 --output-dir ./agents
```

**Available Categories:**
`finance`, `research`, `coding`, `content`, `data-analysis`, `automation`, `customer-service`, `healthcare`, `legal`, `marketing`, `education`, `general`

**Prerequisites:**
- Set `SWARMS_API_KEY` environment variable
- Get your API key at: [https://swarms.world/platform/api-keys](https://swarms.world/platform/api-keys)

For detailed documentation, see [CLI Marketplace Guide](cli_marketplace_guide.md).

### `features` Command

Display all available CLI features and actions in a comprehensive, formatted table.

This command provides a quick reference to all available features, their categories, descriptions, command syntax, and key parameters.

```bash
swarms features
```

**No arguments required.**

**Example:**
```bash
swarms features
```

**Output Includes:**
- **Main Features Table**: Complete list of all features with:
  - Feature name
  - Category (Setup, Auth, Execution, Creation, etc.)
  - Description
  - Command syntax
  - Key parameters
- **Category Summary**: Overview of features grouped by category with counts
- **Usage Tips**: Quick tips for using the CLI effectively

**Use Cases:**
- Quick reference when exploring CLI capabilities
- Discovering available features
- Understanding command syntax and parameters
- Learning about feature categories

## Error Handling

The CLI provides comprehensive error handling with formatted error messages:

### Error Types

| Error Type | Description | Resolution |
|------------|-------------|------------|
| `FileNotFoundError` | Configuration file not found | Check file path and permissions |
| `ValueError` | Invalid configuration format | Verify YAML/markdown syntax |
| `SwarmCLIError` | Custom CLI-specific errors | Check command arguments and API keys |
| `API Key Error` | Authentication issues | Verify API key configuration |
| `Context Length Error` | Model context exceeded | Reduce input size or use larger model |

### Error Display Format

Errors are displayed in formatted panels with:

- **Error Title**: Clear error identification

- **Error Message**: Detailed error description

- **Help Text**: Suggested resolution steps

- **Color Coding**: Red borders for errors, yellow for warnings


## Examples

### Basic Agent Creation

```bash
# Create a simple agent
swarms agent \
  --name "Code Reviewer" \
  --description "AI code review assistant" \
  --system-prompt "You are an expert code reviewer..." \
  --task "Review this Python code for best practices" \
  --model-name "gpt-4" \
  --temperature 0.1
```

### Loading Multiple Agents

```bash
# Load agents from markdown directory
swarms load-markdown \
  --markdown-path ./my_agents/ \
  --concurrent
```

### Running YAML Configuration

```bash
# Execute agents from YAML file
swarms run-agents \
  --yaml-file production_agents.yaml
```

### Autonomous Swarm Generation

```bash
# Generate swarm for complex task
swarms autoswarm \
  --task "Create a comprehensive market analysis report for tech stocks" \
  --model "gpt-4"
```

### LLM Council Collaboration

```bash
# Run LLM Council for collaborative problem solving
swarms llm-council \
  --task "What are the best strategies for reducing carbon emissions in manufacturing?" \
  --verbose
```

### HeavySwarm Complex Analysis

```bash
# Run HeavySwarm for comprehensive task analysis
swarms heavy-swarm \
  --task "Analyze the impact of AI on the job market in 2024" \
  --loops-per-agent 2 \
  --question-agent-model-name "gpt-4" \
  --worker-model-name "gpt-4" \
  --verbose
```

### Viewing All Features

```bash
# Display all available features
swarms features
```

## Configuration

### YAML Configuration Format

For `run-agents` command, use this YAML structure:

```yaml
agents:
  - name: "Research Agent"

    description: "Research and analysis specialist"
    model_name: "gpt-4"
    system_prompt: "You are a research specialist..."
    temperature: 0.1
    max_loops: 3
    
  - name: "Analysis Agent"

    description: "Data analysis expert"
    model_name: "gpt-4"
    system_prompt: "You are a data analyst..."
    temperature: 0.2
    max_loops: 5
```

### Markdown Configuration Format

For `load-markdown` command, use YAML frontmatter:

```markdown
---
name: Research Agent
description: AI research specialist
model_name: gpt-4
temperature: 0.1
max_loops: 3
---

You are an expert research agent specializing in...
```

## Advanced Features

### Progress Indicators

The CLI provides rich progress indicators for long-running operations:

- **Spinner Animations**: Visual feedback during execution


- **Progress Bars**: For operations with known completion states

- **Status Updates**: Real-time operation status


### Concurrent Processing

Multiple markdown files can be processed concurrently:

- **Parallel Execution**: Improves performance for large directories

- **Resource Management**: Automatic thread management

- **Error Isolation**: Individual file failures don't affect others


### Auto-upgrade System

```bash
swarms auto-upgrade
```

Automatically updates Swarms to the latest version with:

- Version checking

- Dependency resolution

- Safe update process


### Interactive Onboarding

```bash
swarms onboarding
```

Guided setup process including:

- API key configuration

- Environment setup

- Basic agent creation

- Usage examples

### Multi-Agent Collaboration

The CLI supports advanced multi-agent architectures:

#### LLM Council

Collaborative problem-solving with multiple specialized models:

```bash
swarms llm-council --task "Your question here"
```

**Features:**
- Multiple model perspectives (GPT-5.1, Gemini, Claude, Grok)
- Anonymous peer review and ranking
- Synthesized final responses
- Cross-model evaluation

#### HeavySwarm

Complex task analysis with specialized agent roles:

```bash
swarms heavy-swarm --task "Your complex task here"
```

**Features:**
- Task decomposition into specialized questions
- Parallel agent execution
- Result synthesis and integration
- Iterative refinement with multiple loops
- Specialized agent roles (Research, Analysis, Writing, Question)

### Feature Discovery

Quickly discover all available features:

```bash
swarms features
```

Displays comprehensive tables showing:
- All available commands
- Feature categories
- Command syntax
- Key parameters
- Usage examples


## Troubleshooting

### Common Issues

1. **API Key Not Set**

   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. **File Permissions**
   ```bash
   chmod 644 agents.yaml
   ```

3. **Model Not Available**
   - Verify model name spelling

   - Check API key permissions

   - Ensure sufficient quota


### Debug Mode

Enable verbose output for debugging:

```bash
swarms <command> --verbose
```

## Integration

### CI/CD Integration

The CLI can be integrated into CI/CD pipelines:

```yaml
# GitHub Actions example
- name: Run Swarms Agents

  run: |
    swarms run-agents --yaml-file ci_agents.yaml
```

### Scripting

Use in shell scripts:

```bash
#!/bin/bash
# Run multiple agent configurations
swarms run-agents --yaml-file agents1.yaml
swarms run-agents --yaml-file agents2.yaml
```

## Performance Considerations

| Consideration           | Recommendation                                      |
|------------------------|-----------------------------------------------------|
| Concurrent Processing  | Use `--concurrent` for multiple files               |
| Model Selection        | Choose appropriate models for task complexity        |
| Context Length         | Monitor and optimize input sizes                    |
| Rate Limiting          | Respect API provider limits                         |
| Multi-Agent Execution  | LLM Council and HeavySwarm execute agents in parallel for efficiency |
| Loop Configuration     | Adjust `--loops-per-agent` based on task complexity and time constraints |

## Security

| Security Aspect         | Recommendation                                         |
|------------------------|--------------------------------------------------------|
| API Key Management     | Store keys in environment variables                    |
| File Permissions       | Restrict access to configuration files                 |
| Input Validation       | CLI validates all inputs before execution              |
| Error Sanitization     | Sensitive information is not exposed in errors         |

## Command Quick Reference

### Quick Start Commands

```bash
# Environment setup
swarms setup-check --verbose
swarms onboarding

# View all features
swarms features

# Get help
swarms help
```

### Agent Commands

```bash
# Create custom agent
swarms agent --name "Agent" --task "Task" --system-prompt "Prompt"

# Run agents from YAML
swarms run-agents --yaml-file agents.yaml

# Load from markdown
swarms load-markdown --markdown-path ./agents/
```

### Multi-Agent Commands

```bash
# LLM Council
swarms llm-council --task "Your question"

# HeavySwarm
swarms heavy-swarm --task "Your complex task" --loops-per-agent 2 --verbose

# Auto-generate swarm
swarms autoswarm --task "Task description" --model "gpt-4"
```

## Support

For additional support:

| Support Option        | Link                                                                                  |
|----------------------|---------------------------------------------------------------------------------------|
| **Community**        | [Discord](https://discord.gg/EamjgSaEQf)                                              |
| **Issues**           | [GitHub Issues](https://github.com/kyegomez/swarms/issues)                            |
| **Strategy Sessions**| [Book a Call](https://cal.com/swarms/swarms-strategy-session)                         |
| **Documentation**    | [Full Documentation](https://docs.swarms.world)                                      |
