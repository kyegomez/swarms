# Swarms CLI Examples

This directory contains shell script examples demonstrating all available Swarms CLI commands and features. Each script is simple, focused, and demonstrates a single CLI command.

## Quick Start

All scripts are executable. Run them directly:

```bash
chmod +x *.sh
./01_setup_check.sh
```

Or execute with bash:

```bash
bash 01_setup_check.sh
```

## Available Examples

### Setup & Configuration

- **[01_setup_check.sh](examples/cli/01_setup_check.sh)** - Environment setup verification
  ```bash
  swarms setup-check
  ```

- **[02_onboarding.sh](examples/cli/02_onboarding.sh)** - Interactive onboarding process
  ```bash
  swarms onboarding
  ```

- **[03_get_api_key.sh](examples/cli/03_get_api_key.sh)** - Retrieve API keys
  ```bash
  swarms get-api-key
  ```

- **[04_check_login.sh](examples/cli/04_check_login.sh)** - Verify authentication
  ```bash
  swarms check-login
  ```

### Agent Management

- **[05_create_agent.sh](examples/cli/05_create_agent.sh)** - Create and run custom agents
  ```bash
  swarms agent --name "Agent" --description "Description" --system-prompt "Prompt" --task "Task"
  ```

- **[06_run_agents_yaml.sh](examples/cli/06_run_agents_yaml.sh)** - Execute agents from YAML
  ```bash
  swarms run-agents --yaml-file agents.yaml
  ```

- **[07_load_markdown.sh](examples/cli/07_load_markdown.sh)** - Load agents from markdown files
  ```bash
  swarms load-markdown --markdown-path ./agents/
  ```

### Multi-Agent Architectures

- **[08_llm_council.sh](examples/cli/08_llm_council.sh)** - Run LLM Council collaboration
  ```bash
  swarms llm-council --task "Your question here"
  ```

- **[09_heavy_swarm.sh](examples/cli/09_heavy_swarm.sh)** - Run HeavySwarm for complex tasks
  ```bash
  swarms heavy-swarm --task "Your complex task here"
  ```

- **[10_autoswarm.sh](examples/cli/10_autoswarm.sh)** - Auto-generate swarm configurations
  ```bash
  swarms autoswarm --task "Task description" --model "gpt-4"
  ```

### Utilities

- **[11_features.sh](examples/cli/11_features.sh)** - Display all available features
  ```bash
  swarms features
  ```

- **[12_help.sh](examples/cli/12_help.sh)** - Display help documentation
  ```bash
  swarms help
  ```

- **[13_auto_upgrade.sh](examples/cli/13_auto_upgrade.sh)** - Update Swarms package
  ```bash
  swarms auto-upgrade
  ```

- **[14_book_call.sh](examples/cli/14_book_call.sh)** - Schedule strategy session
  ```bash
  swarms book-call
  ```

- **[research_agent_example.sh](examples/cli/research_agent_example.sh)** - Research agent example
  ```bash
  swarms research-agent --task "Research task"
  ```

### Run All Examples

- **[run_all_examples.sh](examples/cli/run_all_examples.sh)** - Run multiple examples in sequence
  ```bash
  bash run_all_examples.sh
  ```

## Script Structure

Each script follows a simple pattern:

1. **Shebang** - `#!/bin/bash`
2. **Comment** - Brief description of what the script does
3. **Single Command** - One CLI command execution

Example:
```bash
#!/bin/bash

# Swarms CLI - Setup Check Example
# Verify your Swarms environment setup

swarms setup-check
```

## Usage Patterns

### Basic Command Execution

```bash
swarms <command> [options]
```

### With Verbose Output

```bash
swarms <command> --verbose
```

### Environment Variables

Set API keys before running scripts that require them:

```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
```

## Examples by Category

### Setup & Diagnostics
- Environment setup verification
- Onboarding workflow
- API key management
- Authentication verification

### Single Agent Operations
- Custom agent creation
- Agent configuration from YAML
- Agent loading from markdown

### Multi-Agent Operations
- LLM Council for collaborative problem-solving
- HeavySwarm for complex analysis
- Auto-generated swarm configurations

### Information & Help
- Feature discovery
- Help documentation
- Package management

## File Paths

All scripts are located in `examples/cli/`:

- `examples/cli/01_setup_check.sh`
- `examples/cli/02_onboarding.sh`
- `examples/cli/03_get_api_key.sh`
- `examples/cli/04_check_login.sh`
- `examples/cli/05_create_agent.sh`
- `examples/cli/06_run_agents_yaml.sh`
- `examples/cli/07_load_markdown.sh`
- `examples/cli/08_llm_council.sh`
- `examples/cli/09_heavy_swarm.sh`
- `examples/cli/10_autoswarm.sh`
- `examples/cli/11_features.sh`
- `examples/cli/12_help.sh`
- `examples/cli/13_auto_upgrade.sh`
- `examples/cli/14_book_call.sh`
- `examples/cli/research_agent_example.sh`
- `examples/cli/run_all_examples.sh`

## Related Documentation

- [CLI Reference](../../docs/swarms/cli/cli_reference.md) - Complete CLI documentation
- [Main Examples README](../README.md) - Other Swarms examples
- [Swarms Documentation](../../docs/) - Full Swarms documentation

