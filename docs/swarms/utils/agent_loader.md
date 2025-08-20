# AgentLoader - Load Agents from Markdown Files

The `AgentLoader` is a powerful utility for creating Swarms agents from markdown files using the Claude Code sub-agent format. It supports both single and multiple markdown file loading, providing a flexible way to define and deploy agents using YAML frontmatter configuration.

## Overview

The AgentLoader enables you to:
- Load single agents from markdown files with YAML frontmatter
- Load multiple agents from directories or file lists with concurrent processing
- Parse Claude Code sub-agent YAML frontmatter configurations
- Extract system prompts from markdown content
- Utilize 100% CPU cores for high-performance batch loading
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

# Load agent from markdown file
agent = load_agent_from_markdown("finance_advisor.md")

# Use the agent
response = agent.run(
    "I have $10,000 to invest. What's a good strategy for a beginner?"
)
```

### Loading Multiple Agents (Concurrent)

```python
from swarms.utils import load_agents_from_markdown

# Load agents from list of files with concurrent processing
agents = load_agents_from_markdown([
    "market_researcher.md",
    "financial_analyst.md", 
    "risk_analyst.md"
], concurrent=True)  # Uses all CPU cores for faster loading

# Use agents in a workflow
from swarms.structs import SequentialWorkflow

workflow = SequentialWorkflow(
    agents=agents,
    max_loops=1
)

task = "Analyze the AI healthcare market for a $50M investment."
result = workflow.run(task)
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

# Load multiple agents with concurrent processing
agents = loader.load_multiple_agents(
    "./agents_directory/", 
    concurrent=True,      # Enable concurrent processing
    max_workers=8         # Optional: limit worker threads
)

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

### Example: Finance Advisor Agent

Create a file `finance_advisor.md`:

```markdown
---
name: FinanceAdvisor
description: Expert financial advisor for investment and budgeting guidance
model_name: gpt-4
temperature: 0.7
max_loops: 1
---

You are an expert financial advisor with deep knowledge in:
- Investment strategies and portfolio management
- Personal budgeting and financial planning
- Risk assessment and diversification
- Tax optimization strategies
- Retirement planning

Your approach:
- Provide clear, actionable financial advice
- Consider individual risk tolerance and goals
- Explain complex concepts in simple terms
- Always emphasize the importance of diversification
- Include relevant disclaimers about financial advice

When analyzing financial situations:
1. Assess current financial position
2. Identify short-term and long-term goals
3. Evaluate risk tolerance
4. Recommend appropriate strategies
5. Suggest specific action steps
```

### Loading and Using the Agent

```python
from swarms.utils import load_agent_from_markdown

# Load the Finance Advisor agent
agent = load_agent_from_markdown("finance_advisor.md")

# Use the agent for financial advice
response = agent.run(
    "I have $10,000 to invest. What's a good strategy for a beginner?"
)
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

## Concurrent Processing Features

### Multi-Core Performance

The AgentLoader utilizes **100% of CPU cores** for concurrent agent loading, providing significant performance improvements when processing multiple markdown files:

```python
from swarms.utils import load_agents_from_markdown

# Automatic concurrent processing for multiple files
agents = load_agents_from_markdown([
    "agent1.md", "agent2.md", "agent3.md", "agent4.md"
])  # concurrent=True by default

# Manual control over concurrency
agents = load_agents_from_markdown(
    "./agents_directory/",
    concurrent=True,        # Enable concurrent processing
    max_workers=8           # Limit to 8 worker threads
)

# Disable concurrency for debugging or single files
agents = load_agents_from_markdown(
    ["single_agent.md"],
    concurrent=False        # Sequential processing
)
```

### Resource Management

```python
# Default: Uses all CPU cores
agents = load_agents_from_markdown(files, concurrent=True)

# Custom worker count for resource control
agents = load_agents_from_markdown(
    files, 
    concurrent=True,
    max_workers=4  # Limit to 4 threads
)

# ThreadPoolExecutor automatically manages:
# - Thread lifecycle
# - Resource cleanup
# - Exception handling
# - Result collection
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
def load_agent_from_markdown(file_path: str, **kwargs) -> Agent
def load_agents_from_markdown(
    file_paths: Union[str, List[str]], 
    concurrent: bool = True,          # Enable concurrent processing
    max_workers: Optional[int] = None, # Max worker threads (defaults to CPU count)
    **kwargs
) -> List[Agent]
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
- `agents_loader_example.py` - Simple usage example
- `examples/agent_loader_demo.py` - Multi-agent workflow example

## Support

For questions and support:
- GitHub Issues: [https://github.com/kyegomez/swarms/issues](https://github.com/kyegomez/swarms/issues)
- Documentation: [https://docs.swarms.world](https://docs.swarms.world)
- Community: Join our Discord for real-time support