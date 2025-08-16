# AgentLoader - Load Agents from Markdown Files

The `AgentLoader` is a powerful utility for creating Swarms agents from markdown files using the Claude Code sub-agent format. It supports both single and multiple markdown file loading, providing a flexible way to define and deploy agents using YAML frontmatter configuration.

## Overview

The AgentLoader enables you to:
- Load single agents from markdown files with YAML frontmatter
- Load multiple agents from directories or file lists
- Parse Claude Code sub-agent YAML frontmatter configurations
- Extract system prompts from markdown content
- Provide comprehensive error handling and validation

## Installation

The AgentLoader is included with the Swarms framework:

```python
from swarms.utils import AgentLoader, load_agent_from_markdown, load_agents_from_markdown
```

## Markdown Format

The AgentLoader uses the Claude Code sub-agent YAML frontmatter format:

```markdown
---
name: your-sub-agent-name
description: Description of when this subagent should be invoked
model_name: gpt-4
temperature: 0.3
max_loops: 2
mcp_url: http://example.com/mcp  # optional
---

Your subagent's system prompt goes here. This can be multiple paragraphs
and should clearly define the subagent's role, capabilities, and approach
to solving problems.

Include specific instructions, best practices, and any constraints
the subagent should follow.
```

**Schema Fields:**
- `name` (required): Your sub-agent name
- `description` (required): Description of when this subagent should be invoked
- `model_name` (optional): Name of model (defaults to random selection if not provided)
- `temperature` (optional): Float value for model temperature (0.0-2.0)
- `max_loops` (optional): Integer for maximum reasoning loops
- `mcp_url` (optional): MCP server URL if needed

## Quick Start

### Loading a Single Agent

```python
from swarms.utils import load_agent_from_markdown

# Load Claude Code format agent (YAML frontmatter)
agent = load_agent_from_markdown(
    file_path="performance-engineer.md"  # Uses YAML frontmatter format
)

# The agent automatically gets configured with:
# - Name, description from frontmatter
# - Temperature, max_loops, model settings
# - System prompt from content after frontmatter

response = agent.run("Analyze application performance issues")
print(response)
```

### Loading Multiple Agents

```python
from swarms.utils import load_agents_from_markdown

# Load all agents from directory (YAML frontmatter format)
agents = load_agents_from_markdown(
    file_paths="./agents_directory/"  # Directory with Claude Code format files
)

# Load agents from specific files
agents = load_agents_from_markdown(
    file_paths=[
        "performance-engineer.md",  # Claude Code YAML format
        "financial-analyst.md",     # Claude Code YAML format
        "security-analyst.md"       # Claude Code YAML format
    ]
)

print(f"Loaded {len(agents)} agents")
for agent in agents:
    print(f"- {agent.agent_name}: {getattr(agent, 'temperature', 'default temp')}")
```

## Class-Based Usage

### AgentLoader Class

For more advanced usage, use the `AgentLoader` class directly:

```python
from swarms.utils import AgentLoader

# Initialize loader
loader = AgentLoader()

# Load single agent
agent = loader.load_single_agent("path/to/agent.md")

# Load multiple agents
agents = loader.load_multiple_agents("./agents_directory/")

# Parse markdown file without creating agent
config = loader.parse_markdown_file("path/to/agent.md")
print(config.name, config.description)
```

## Configuration Options

You can override default configuration when loading agents:

```python
agent = load_agent_from_markdown(
    file_path="agent.md",
    max_loops=5,
    verbose=True,
    dashboard=True,
    autosave=False,
    context_length=200000
)
```

### Available Configuration Parameters

- `max_loops` (int): Maximum number of reasoning loops (default: 1)
- `autosave` (bool): Enable automatic state saving (default: True)
- `dashboard` (bool): Enable dashboard monitoring (default: False)
- `verbose` (bool): Enable verbose logging (default: False)
- `dynamic_temperature_enabled` (bool): Enable dynamic temperature (default: False)
- `saved_state_path` (str): Path for saving agent state
- `user_name` (str): User identifier (default: "default_user")
- `retry_attempts` (int): Number of retry attempts (default: 3)
- `context_length` (int): Maximum context length (default: 100000)
- `return_step_meta` (bool): Return step metadata (default: False)
- `output_type` (str): Output format type (default: "str")
- `auto_generate_prompt` (bool): Auto-generate prompts (default: False)
- `artifacts_on` (bool): Enable artifacts (default: False)

## Complete Example

### Example: Claude Code Sub-Agent Format

Create a file `performance-engineer.md`:

```markdown
---
name: performance-engineer
description: Optimize application performance and identify bottlenecks
model_name: gpt-4
temperature: 0.3
max_loops: 2
mcp_url: http://example.com/mcp
---

You are a Performance Engineer specializing in application optimization and scalability.

Your role involves analyzing system performance, identifying bottlenecks, and implementing 
solutions to improve efficiency and user experience.

Key responsibilities:
- Profile applications to identify performance issues
- Optimize database queries and caching strategies  
- Implement load testing and monitoring solutions
- Recommend infrastructure improvements
- Provide actionable optimization recommendations

Always provide specific, measurable recommendations with implementation details.
Focus on both immediate wins and long-term architectural improvements.
```

### Loading and Using the Agent

```python
from swarms.utils import load_agent_from_markdown

# Load Claude Code format agent (YAML frontmatter)
performance_agent = load_agent_from_markdown(
    file_path="performance-engineer.md"
)

print(f"Agent: {performance_agent.agent_name}")
print(f"Temperature: {getattr(performance_agent, 'temperature', 'default')}")
print(f"Max loops: {performance_agent.max_loops}")
print(f"System prompt preview: {performance_agent.system_prompt[:100]}...")

# Use the performance agent
task = """
Analyze the performance of a web application that handles 10,000 concurrent users
but is experiencing slow response times averaging 3 seconds. The application uses
a PostgreSQL database and is deployed on AWS with 4 EC2 instances behind a load balancer.
"""

# Note: Actual agent.run() would make API calls
print(f"\nTask for {performance_agent.agent_name}: {task[:100]}...")
```

## Error Handling

The AgentLoader provides comprehensive error handling:

```python
from swarms.utils import AgentLoader

loader = AgentLoader()

try:
    # This will raise FileNotFoundError
    agent = loader.load_single_agent("nonexistent.md")
except FileNotFoundError as e:
    print(f"File not found: {e}")

try:
    # This will handle parsing errors gracefully
    agents = loader.load_multiple_agents("./invalid_directory/")
    print(f"Successfully loaded {len(agents)} agents")
except Exception as e:
    print(f"Error loading agents: {e}")
```

## Advanced Features

### Custom System Prompt Building

The AgentLoader automatically builds comprehensive system prompts from the markdown structure:

```python
loader = AgentLoader()
config = loader.parse_markdown_file("agent.md")

# The system prompt includes:
# - Role description from the table
# - Focus areas as bullet points
# - Approach as numbered steps
# - Expected outputs as deliverables

print("Generated System Prompt:")
print(config.system_prompt)
```

### Batch Processing

Process multiple agent files efficiently:

```python
import os
from pathlib import Path
from swarms.utils import AgentLoader

loader = AgentLoader()

# Find all markdown files in a directory
agent_dir = Path("./agents")
md_files = list(agent_dir.glob("*.md"))

# Load all agents
agents = []
for file_path in md_files:
    try:
        agent = loader.load_single_agent(str(file_path))
        agents.append(agent)
        print(f"✓ Loaded: {agent.agent_name}")
    except Exception as e:
        print(f"✗ Failed to load {file_path}: {e}")

print(f"\nSuccessfully loaded {len(agents)} agents")
```

## Integration with Swarms

The loaded agents are fully compatible with Swarms orchestration systems:

```python
from swarms.utils import load_agents_from_markdown
from swarms.structs import SequentialWorkflow

# Load multiple specialized agents
agents = load_agents_from_markdown("./specialist_agents/")

# Create a sequential workflow
workflow = SequentialWorkflow(
    agents=agents,
    max_loops=1
)

# Execute complex task across multiple agents
result = workflow.run("Conduct a comprehensive system audit")
```

## Best Practices

1. **Consistent Naming**: Use clear, descriptive agent names
2. **Detailed Descriptions**: Provide comprehensive role descriptions
3. **Structured Sections**: Use the optional sections to define agent behavior
4. **Error Handling**: Always wrap agent loading in try-catch blocks
5. **Model Selection**: Choose appropriate models based on agent complexity
6. **Configuration**: Override defaults when specific behavior is needed


## API Reference

### AgentLoader Class

```python
class AgentLoader:
    def __init__(self, model: Optional[LiteLLM] = None)
    def parse_markdown_file(self, file_path: str) -> MarkdownAgentConfig
    def load_single_agent(self, file_path: str, **kwargs) -> Agent
    def load_multiple_agents(self, file_paths: Union[str, List[str]], **kwargs) -> List[Agent]
```

### Convenience Functions

```python
def load_agent_from_markdown(file_path: str, model: Optional[LiteLLM] = None, **kwargs) -> Agent
def load_agents_from_markdown(file_paths: Union[str, List[str]], model: Optional[LiteLLM] = None, **kwargs) -> List[Agent]
```

### Configuration Model

```python
class MarkdownAgentConfig(BaseModel):
    name: str
    description: str
    model_name: Optional[str] = "gpt-4"
    temperature: Optional[float] = 0.1  # Model temperature (0.0-2.0)
    mcp_url: Optional[str] = None       # Optional MCP server URL
    system_prompt: str
    max_loops: int = 1
    autosave: bool = False
    dashboard: bool = False
    verbose: bool = False
    # ... additional configuration fields
```

## Examples Repository

Find more examples in the Swarms repository:
- `examples/agents_loader_example.py` - Complete usage demonstration
- `test_agent_loader.py` - Test suite with validation examples
- `examples/single_agent/utils/markdown_agent.py` - Markdown agent utilities

## Support

For questions and support:
- GitHub Issues: [https://github.com/kyegomez/swarms/issues](https://github.com/kyegomez/swarms/issues)
- Documentation: [https://docs.swarms.world](https://docs.swarms.world)
- Community: Join our Discord for real-time support