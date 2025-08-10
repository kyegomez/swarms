# AgentLoader - Load Agents from Markdown Files

The `AgentLoader` is a powerful utility for creating Swarms agents from markdown files. It supports both single and multiple markdown file loading, providing a flexible way to define and deploy agents using a structured markdown format.

## Overview

The AgentLoader enables you to:
- Load single agents from markdown files
- Load multiple agents from directories or file lists
- Parse structured markdown content into agent configurations
- Maintain backwards compatibility with existing agent systems
- Provide comprehensive error handling and validation

## Installation

The AgentLoader is included with the Swarms framework:

```python
from swarms.utils import AgentLoader, load_agent_from_markdown, load_agents_from_markdown
```

## Markdown Format

The AgentLoader expects markdown files to follow a specific structure:

### Required Table Header
```markdown
| name | description | model |
|------|-------------|-------|
| agent-name | Brief description of the agent | gpt-4 |
```

### Optional Sections
```markdown
## Focus Areas
- Key responsibility area 1
- Key responsibility area 2
- Key responsibility area 3

## Approach
1. First step in methodology
2. Second step in methodology
3. Third step in methodology

## Output
- Expected deliverable 1
- Expected deliverable 2
- Expected deliverable 3
```

## Quick Start

### Loading a Single Agent

```python
from swarms.utils import load_agent_from_markdown

# Load agent from markdown file
agent = load_agent_from_markdown(
    file_path="path/to/agent.md"
)

# Use the agent
response = agent.run("What are your capabilities?")
print(response)
```

### Loading Multiple Agents

```python
from swarms.utils import load_agents_from_markdown

# Load all agents from directory
agents = load_agents_from_markdown(
    file_paths="./agents_directory/"
)

# Load agents from specific files
agents = load_agents_from_markdown(
    file_paths=["agent1.md", "agent2.md", "agent3.md"]
)

print(f"Loaded {len(agents)} agents")
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

### Example Markdown File (performance-engineer.md)

```markdown
| name | description | model |
|------|-------------|-------|
| performance-engineer | Optimize application performance and identify bottlenecks | gpt-4 |

## Focus Areas
- Application profiling and performance analysis
- Database optimization and query tuning
- Memory and CPU usage optimization
- Load testing and capacity planning
- Infrastructure scaling recommendations

## Approach
1. Analyze application architecture and identify potential bottlenecks
2. Implement comprehensive monitoring and logging systems
3. Conduct performance testing under various load conditions
4. Profile memory usage and optimize resource consumption
5. Provide actionable recommendations with implementation guides

## Output
- Detailed performance analysis reports with metrics
- Optimized code recommendations and examples
- Infrastructure scaling and architecture suggestions
- Monitoring and alerting configuration guidelines
- Load testing results and capacity planning documents
```

### Loading and Using the Agent

```python
from swarms.utils import load_agent_from_markdown
from swarms.utils.litellm_wrapper import LiteLLM

# Initialize model
model = LiteLLM(model_name="gpt-4")

# Load the performance engineer agent
agent = load_agent_from_markdown(
    file_path="performance-engineer.md",
    model=model,
    max_loops=3,
    verbose=True
)

# Use the agent
task = """
Analyze the performance of a web application that handles 10,000 concurrent users
but is experiencing slow response times averaging 3 seconds. The application uses
a PostgreSQL database and is deployed on AWS with 4 EC2 instances behind a load balancer.
"""

analysis = agent.run(task)
print(f"Performance Analysis:\n{analysis}")
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

## Backwards Compatibility

The AgentLoader maintains full backwards compatibility with:
- Claude Code sub-agents markdown format
- Existing swarms agent creation patterns
- Legacy configuration systems
- Current workflow orchestration

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
    system_prompt: str
    focus_areas: Optional[List[str]] = []
    approach: Optional[List[str]] = []
    output: Optional[List[str]] = []
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