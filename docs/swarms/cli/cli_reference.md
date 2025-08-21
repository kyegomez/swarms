# Swarms CLI Reference

The Swarms CLI is a comprehensive command-line interface for managing and executing Swarms agents and multi-agent architectures. This reference documents all available commands, arguments, and features.

## Table of Contents

- [Installation](#installation)

- [Basic Usage](#basic-usage)

- [Commands Reference](#commands-reference)

- [Global Arguments](#global-arguments)

- [Command-Specific Arguments](#command-specific-arguments)

- [Error Handling](#error-handling)

- [Examples](#examples)

- [Configuration](#configuration)


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
| `get-api-key` | Open API key portal in browser | None |
| `check-login` | Verify login status and initialize cache | None |
| `run-agents` | Execute agents from YAML configuration | `--yaml-file` |
| `load-markdown` | Load agents from markdown files | `--markdown-path` |
| `agent` | Create and run custom agent | `--name`, `--description`, `--system-prompt`, `--task` |
| `auto-upgrade` | Update Swarms to latest version | None |
| `book-call` | Schedule strategy session | None |
| `autoswarm` | Generate and execute autonomous swarm | `--task`, `--model` |
| `setup-check` | Run comprehensive environment setup check | None |

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

## Security

| Security Aspect         | Recommendation                                         |
|------------------------|--------------------------------------------------------|
| API Key Management     | Store keys in environment variables                    |
| File Permissions       | Restrict access to configuration files                 |
| Input Validation       | CLI validates all inputs before execution              |
| Error Sanitization     | Sensitive information is not exposed in errors         |

## Support

For additional support:

| Support Option        | Link                                                                                  |
|----------------------|---------------------------------------------------------------------------------------|
| **Community**        | [Discord](https://discord.gg/EamjgSaEQf)                                              |
| **Issues**           | [GitHub Issues](https://github.com/kyegomez/swarms/issues)                            |
| **Strategy Sessions**| [Book a Call](https://cal.com/swarms/swarms-strategy-session)                         |
