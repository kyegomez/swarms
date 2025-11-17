# AgentLoader Documentation

The `AgentLoader` is a comprehensive utility for creating Swarms agents from various file formats including Markdown, YAML, and CSV files. It provides a unified interface for loading agents with support for concurrent processing, configuration overrides, and automatic file type detection.

## Overview

The AgentLoader enables you to:

- Load agents from Markdown files 
- Load agents from YAML configuration files
- Load agents from CSV files
- Automatically detect file types and use appropriate loaders
- Process multiple files concurrently for improved performance
- Override default configurations with custom parameters
- Handle various agent configurations and settings

## Installation

The AgentLoader is included with the Swarms framework:

```python
from swarms.structs import AgentLoader
from swarms.utils import load_agent_from_markdown, load_agents_from_markdown
```

## Supported File Formats

### 1. Markdown Files (Claude Code Format)

The primary format uses YAML frontmatter with markdown content:

```markdown
---
name: FinanceAdvisor
description: Expert financial advisor for investment and budgeting guidance
model_name: claude-sonnet-4-20250514
temperature: 0.7
max_loops: 1
mcp_url: http://example.com/mcp  # optional
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

**Schema Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | string | ✅ Yes | - | Your agent name |
| `description` | string | ✅ Yes | - | Description of the agent's role and capabilities |
| `model_name` | string | ❌ No | "gpt-4.1" | Name of the model to use |
| `temperature` | float | ❌ No | 0.1 | Model temperature (0.0-2.0) |
| `max_loops` | integer | ❌ No | 1 | Maximum reasoning loops |
| `mcp_url` | string | ❌ No | None | MCP server URL if needed |
| `streaming_on` | boolean | ❌ No | False | Enable streaming output |

### 2. YAML Files

YAML configuration files for agent definitions:

```yaml
agents:
  - name: "ResearchAgent"
    description: "Research and analysis specialist"
    model_name: "gpt-4"
    temperature: 0.3
    max_loops: 2
    system_prompt: "You are a research specialist..."
```

### 3. CSV Files

CSV files with agent configurations:

```csv
name,description,model_name,temperature,max_loops
ResearchAgent,Research specialist,gpt-4,0.3,2
AnalysisAgent,Data analyst,claude-3,0.1,1
```

## Quick Start

### Loading a Single Agent

```python
from swarms.structs import AgentLoader

# Initialize the loader
loader = AgentLoader()

# Load agent from markdown file
agent = loader.load_agent_from_markdown("finance_advisor.md")

# Use the agent
response = agent.run(
    "I have $10,000 to invest. What's a good strategy for a beginner?"
)
```

### Loading Multiple Agents

```python
from swarms.structs import AgentLoader

loader = AgentLoader()

# Load agents from list of files with concurrent processing
agents = loader.load_agents_from_markdown([
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

### Automatic File Type Detection

```python
from swarms.structs import AgentLoader

loader = AgentLoader()

# Automatically detect file type and load appropriately
agents = loader.auto("agents.yaml")  # YAML file
agents = loader.auto("agents.csv")    # CSV file
agents = loader.auto("agents.md")     # Markdown file
```

## Class-Based Usage

### AgentLoader Class

For more advanced usage, use the `AgentLoader` class directly:

```python
from swarms.structs import AgentLoader

# Initialize loader
loader = AgentLoader()

# Load single agent
agent = loader.load_single_agent("path/to/agent.md")

# Load multiple agents with concurrent processing
agents = loader.load_multiple_agents(
    "./agents_directory/", 
    concurrent=True,      # Enable concurrent processing
    max_file_size_mb=10.0 # Limit file size for memory safety
)

# Parse markdown file without creating agent
config = loader.parse_markdown_file("path/to/agent.md")
print(config.name, config.description)
```

## Configuration Options

You can override default configuration when loading agents:

```python
agent = loader.load_agent_from_markdown(
    file_path="agent.md",
    max_loops=5,
    verbose=True,
    dashboard=True,
    autosave=False,
    context_length=200000,
    temperature=0.5
)
```

### Available Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_loops` | int | 1 | Maximum number of reasoning loops |
| `autosave` | bool | False | Enable automatic state saving |
| `dashboard` | bool | False | Enable dashboard monitoring |
| `verbose` | bool | False | Enable verbose logging |
| `dynamic_temperature_enabled` | bool | False | Enable dynamic temperature |
| `saved_state_path` | str | None | Path for saving agent state |
| `user_name` | str | "default_user" | User identifier |
| `retry_attempts` | int | 3 | Number of retry attempts |
| `context_length` | int | 100000 | Maximum context length |
| `return_step_meta` | bool | False | Return step metadata |
| `output_type` | str | "str" | Output format type |
| `auto_generate_prompt` | bool | False | Auto-generate prompts |
| `streaming_on` | bool | False | Enable streaming output |
| `mcp_url` | str | None | MCP server URL if needed |

## Advanced Features

### Concurrent Processing

The AgentLoader utilizes multiple CPU cores for concurrent agent loading:

```python
from swarms.structs import AgentLoader

loader = AgentLoader()

# Automatic concurrent processing for multiple files
agents = loader.load_agents_from_markdown([
    "agent1.md", "agent2.md", "agent3.md", "agent4.md"
])  # concurrent=True by default

# Manual control over concurrency
agents = loader.load_agents_from_markdown(
    "./agents_directory/",
    concurrent=True,        # Enable concurrent processing
    max_file_size_mb=5.0   # Limit file size for memory safety
)

# Disable concurrency for debugging or single files
agents = loader.load_agents_from_markdown(
    ["single_agent.md"],
    concurrent=False        # Sequential processing
)
```

### File Size Validation

```python
# Set maximum file size to prevent memory issues
agents = loader.load_agents_from_markdown(
    "./agents_directory/",
    max_file_size_mb=5.0  # Skip files larger than 5MB
)
```

### Multiple File Type Support

```python
from swarms.structs import AgentLoader

loader = AgentLoader()

# Load from different file types
yaml_agents = loader.load_agents_from_yaml("agents.yaml")
csv_agents = loader.load_agents_from_csv("agents.csv")
md_agents = loader.load_agents_from_markdown("agents.md")

# Load from multiple YAML files with different return types
yaml_files = ["agents1.yaml", "agents2.yaml"]
return_types = ["auto", "list"]
agents = loader.load_many_agents_from_yaml(yaml_files, return_types)
```

## Complete Examples

### Example 1: Finance Advisor Agent

Create a file `finance_advisor.md`:

```markdown
---
name: FinanceAdvisor
description: Expert financial advisor for investment and budgeting guidance
model_name: claude-sonnet-4-20250514
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
from swarms.structs import AgentLoader

# Load the Finance Advisor agent
loader = AgentLoader()
agent = loader.load_agent_from_markdown("finance_advisor.md")

# Use the agent for financial advice
response = agent.run(
    "I have $10,000 to invest. What's a good strategy for a beginner?"
)
```

### Example 2: Multi-Agent Workflow

```python
from swarms.structs import AgentLoader, SequentialWorkflow

# Load multiple specialized agents
loader = AgentLoader()
agents = loader.load_agents_from_markdown([
    "market_researcher.md",
    "financial_analyst.md", 
    "risk_analyst.md"
], concurrent=True)

# Create a sequential workflow
workflow = SequentialWorkflow(
    agents=agents,
    max_loops=1
)

# Execute complex task across multiple agents
task = """
Analyze the AI healthcare market for a $50M investment opportunity.
Focus on market size, competition, financials, and risks.
"""

result = workflow.run(task)
```

### Example 3: Mixed File Types

```python
from swarms.structs import AgentLoader

loader = AgentLoader()

# Load agents from different file types
markdown_agents = loader.load_agents_from_markdown("./md_agents/")
yaml_agents = loader.load_agents_from_yaml("config.yaml")
csv_agents = loader.load_agents_from_csv("data.csv")

# Combine all agents
all_agents = markdown_agents + yaml_agents + csv_agents

print(f"Loaded {len(all_agents)} agents from various sources")
```

## Error Handling

The AgentLoader provides comprehensive error handling:

```python
from swarms.structs import AgentLoader

loader = AgentLoader()

try:
    # This will raise FileNotFoundError
    agent = loader.load_agent_from_markdown("nonexistent.md")
except FileNotFoundError as e:
    print(f"File not found: {e}")

try:
    # This will handle parsing errors gracefully
    agents = loader.load_multiple_agents("./invalid_directory/")
    print(f"Successfully loaded {len(agents)} agents")
except Exception as e:
    print(f"Error loading agents: {e}")
```

## Best Practices

1. **Consistent Naming**: Use clear, descriptive agent names
2. **Detailed Descriptions**: Provide comprehensive role descriptions
3. **Structured Content**: Use clear sections to define agent behavior
4. **Error Handling**: Always wrap agent loading in try-catch blocks
5. **Model Selection**: Choose appropriate models based on agent complexity
6. **Configuration**: Override defaults when specific behavior is needed
7. **File Organization**: Organize agents by domain or function
8. **Memory Management**: Use `max_file_size_mb` for large agent collections

## API Reference

### AgentLoader Class

```python
class AgentLoader:
    """
    Loader class for creating Agent objects from various file formats.
    
    This class provides methods to load agents from Markdown, YAML, and CSV files.
    """
    
    def __init__(self):
        """Initialize the AgentLoader instance."""
        pass
    
    def load_agents_from_markdown(
        self,
        file_paths: Union[str, List[str]],
        concurrent: bool = True,
        max_file_size_mb: float = 10.0,
        **kwargs
    ) -> List[Agent]:
        """
        Load multiple agents from one or more Markdown files.
        
        Args:
            file_paths: Path or list of paths to Markdown file(s)
            concurrent: Whether to load files concurrently
            max_file_size_mb: Maximum file size in MB to process
            **kwargs: Additional keyword arguments passed to the underlying loader
            
        Returns:
            A list of loaded Agent objects
        """
    
    def load_agent_from_markdown(
        self, 
        file_path: str, 
        **kwargs
    ) -> Agent:
        """
        Load a single agent from a Markdown file.
        
        Args:
            file_path: Path to the Markdown file containing the agent definition
            **kwargs: Additional keyword arguments passed to the underlying loader
            
        Returns:
            The loaded Agent object
        """
    
    def load_agents_from_yaml(
        self,
        yaml_file: str,
        return_type: ReturnTypes = "auto",
        **kwargs
    ) -> List[Agent]:
        """
        Load agents from a YAML file.
        
        Args:
            yaml_file: Path to the YAML file containing agent definitions
            return_type: The return type for the loader
            **kwargs: Additional keyword arguments passed to the underlying loader
            
        Returns:
            A list of loaded Agent objects
        """
    
    def load_agents_from_csv(
        self, 
        csv_file: str, 
        **kwargs
    ) -> List[Agent]:
        """
        Load agents from a CSV file.
        
        Args:
            csv_file: Path to the CSV file containing agent definitions
            **kwargs: Additional keyword arguments passed to the underlying loader
            
        Returns:
            A list of loaded Agent objects
        """
    
    def auto(
        self, 
        file_path: str, 
        *args, 
        **kwargs
    ):
        """
        Automatically load agents from a file based on its extension.
        
        Args:
            file_path: Path to the agent file (Markdown, YAML, or CSV)
            *args: Additional positional arguments passed to the underlying loader
            **kwargs: Additional keyword arguments passed to the underlying loader
            
        Returns:
            A list of loaded Agent objects
            
        Raises:
            ValueError: If the file type is not supported
        """
```

**Method Parameters and Return Types:**

| Method | Parameters | Type | Required | Default | Return Type | Description |
|--------|------------|------|----------|---------|-------------|-------------|
| `load_agents_from_markdown` | `file_paths` | Union[str, List[str]] | ✅ Yes | - | List[Agent] | File path(s) or directory |
| `load_agents_from_markdown` | `concurrent` | bool | ❌ No | True | List[Agent] | Enable concurrent processing |
| `load_agents_from_markdown` | `max_file_size_mb` | float | ❌ No | 10.0 | List[Agent] | Max file size in MB |
| `load_agents_from_markdown` | `**kwargs` | dict | ❌ No | {} | List[Agent] | Configuration overrides |
| `load_agent_from_markdown` | `file_path` | str | ✅ Yes | - | Agent | Path to markdown file |
| `load_agent_from_markdown` | `**kwargs` | dict | ❌ No | {} | Agent | Configuration overrides |
| `load_agents_from_yaml` | `yaml_file` | str | ✅ Yes | - | List[Agent] | Path to YAML file |
| `load_agents_from_yaml` | `return_type` | ReturnTypes | ❌ No | "auto" | List[Agent] | Return type for loader |
| `load_agents_from_yaml` | `**kwargs` | dict | ❌ No | {} | List[Agent] | Configuration overrides |
| `load_agents_from_csv` | `csv_file` | str | ✅ Yes | - | List[Agent] | Path to CSV file |
| `load_agents_from_csv` | `**kwargs` | dict | ❌ No | {} | List[Agent] | Configuration overrides |
| `auto` | `file_path` | str | ✅ Yes | - | List[Agent] | Path to agent file |
| `auto` | `*args` | tuple | ❌ No | () | List[Agent] | Positional arguments |
| `auto` | `**kwargs` | dict | ❌ No | {} | List[Agent] | Keyword arguments |

### Convenience Functions

```python
def load_agent_from_markdown(
    file_path: str, 
    **kwargs
) -> Agent:
    """
    Load a single agent from a markdown file using the Claude Code YAML frontmatter format.
    
    Args:
        file_path: Path to the markdown file containing YAML frontmatter
        **kwargs: Optional keyword arguments to override agent configuration
        
    Returns:
        Configured Agent instance loaded from the markdown file
    """

def load_agents_from_markdown(
    file_paths: Union[str, List[str]],
    concurrent: bool = True,
    max_file_size_mb: float = 10.0,
    **kwargs
) -> List[Agent]:
    """
    Load multiple agents from markdown files using the Claude Code YAML frontmatter format.
    
    Args:
        file_paths: Either a directory path containing markdown files or a list of markdown file paths
        concurrent: If True, enables concurrent processing for faster loading
        max_file_size_mb: Maximum file size (in MB) for each markdown file
        **kwargs: Optional keyword arguments to override agent configuration
        
    Returns:
        List of configured Agent instances loaded from the markdown files
    """
```

**Function Parameters:**

| Function | Parameter | Type | Required | Default | Description |
|----------|-----------|------|----------|---------|-------------|
| `load_agent_from_markdown` | `file_path` | str | ✅ Yes | - | Path to markdown file |
| `load_agent_from_markdown` | `**kwargs` | dict | ❌ No | {} | Configuration overrides |
| `load_agents_from_markdown` | `file_paths` | Union[str, List[str]] | ✅ Yes | - | File path(s) or directory |
| `load_agents_from_markdown` | `concurrent` | bool | ❌ No | True | Enable concurrent processing |
| `load_agents_from_markdown` | `max_file_size_mb` | float | ❌ No | 10.0 | Max file size in MB |
| `load_agents_from_markdown` | `**kwargs` | dict | ❌ No | {} | Configuration overrides |

### Configuration Model

```python
class MarkdownAgentConfig(BaseModel):
    """Configuration model for agents loaded from Claude Code markdown files."""
    
    name: Optional[str] = None
    description: Optional[str] = None
    model_name: Optional[str] = "gpt-4.1"
    temperature: Optional[float] = Field(default=0.1, ge=0.0, le=2.0)
    mcp_url: Optional[int] = None
    system_prompt: Optional[str] = None
    max_loops: Optional[int] = Field(default=1, ge=1)
    autosave: Optional[bool] = False
    dashboard: Optional[bool] = False
    verbose: Optional[bool] = False
    dynamic_temperature_enabled: Optional[bool] = False
    saved_state_path: Optional[str] = None
    user_name: Optional[str] = "default_user"
    retry_attempts: Optional[int] = Field(default=3, ge=1)
    context_length: Optional[int] = Field(default=100000, ge=1000)
    return_step_meta: Optional[bool] = False
    output_type: Optional[str] = "str"
    auto_generate_prompt: Optional[bool] = False
    streaming_on: Optional[bool] = False
```

**MarkdownAgentConfig Schema:**

| Field | Type | Required | Default | Validation | Description |
|-------|------|----------|---------|------------|-------------|
| `name` | Optional[str] | ❌ No | None | - | Agent name |
| `description` | Optional[str] | ❌ No | None | - | Agent description |
| `model_name` | Optional[str] | ❌ No | "gpt-4.1" | - | Model to use |
| `temperature` | Optional[float] | ❌ No | 0.1 | 0.0 ≤ x ≤ 2.0 | Model temperature |
| `mcp_url` | Optional[int] | ❌ No | None | - | MCP server URL |
| `system_prompt` | Optional[str] | ❌ No | None | Non-empty string | System prompt |
| `max_loops` | Optional[int] | ❌ No | 1 | ≥ 1 | Maximum reasoning loops |
| `autosave` | Optional[bool] | ❌ No | False | - | Enable auto-save |
| `dashboard` | Optional[bool] | ❌ No | False | - | Enable dashboard |
| `verbose` | Optional[bool] | ❌ No | False | - | Enable verbose logging |
| `dynamic_temperature_enabled` | Optional[bool] | ❌ No | False | - | Enable dynamic temperature |
| `saved_state_path` | Optional[str] | ❌ No | None | - | State save path |
| `user_name` | Optional[str] | ❌ No | "default_user" | - | User identifier |
| `retry_attempts` | Optional[int] | ❌ No | 3 | ≥ 1 | Retry attempts |
| `context_length` | Optional[int] | ❌ No | 100000 | ≥ 1000 | Context length |
| `return_step_meta` | Optional[bool] | ❌ No | False | - | Return step metadata |
| `output_type` | Optional[str] | ❌ No | "str" | - | Output format |
| `auto_generate_prompt` | Optional[bool] | ❌ No | False | - | Auto-generate prompts |
| `streaming_on` | Optional[bool] | ❌ No | False | - | Enable streaming |

## Examples Repository

Find complete working examples in the `examples/utils/agent_loader/` directory:

### Single Agent Example (`agent_loader_demo.py`)

```python
from swarms.utils import load_agent_from_markdown

agent = load_agent_from_markdown("finance_advisor.md")

agent.run(task="What were the best performing etfs in 2023")
```

### Multi-Agent Workflow Example (`multi_agents_loader_demo.py`)

```python
from swarms.utils import load_agents_from_markdown

agents = load_agents_from_markdown([
    "market_researcher.md",
    "financial_analyst.md", 
    "risk_analyst.md"
])

# Use agents in a workflow
from swarms.structs.sequential_workflow import SequentialWorkflow

workflow = SequentialWorkflow(
    agents=agents,
    max_loops=1
)

task = """
Analyze the AI healthcare market for a $50M investment opportunity.
Focus on market size, competition, financials, and risks.
"""

result = workflow.run(task)
```

### Sample Agent Definition (`finance_advisor.md`)

```markdown
---
name: FinanceAdvisor
description: Expert financial advisor for investment and budgeting guidance
model_name: claude-sonnet-4-20250514
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

## Performance Considerations

### Concurrent Processing

- **Default Behavior**: Uses `os.cpu_count() * 2` worker threads
- **Memory Management**: Automatically validates file sizes before processing
- **Timeout Handling**: 5-minute total timeout, 1-minute per agent timeout
- **Error Recovery**: Continues processing other files if individual files fail

### File Size Limits

- **Default Limit**: 10MB maximum file size
- **Configurable**: Adjustable via `max_file_size_mb` parameter
- **Memory Safety**: Prevents memory issues with large agent definitions

### Resource Optimization

```python
# For large numbers of agents, consider batch processing
loader = AgentLoader()

# Process in smaller batches
batch1 = loader.load_agents_from_markdown("./batch1/", concurrent=True)
batch2 = loader.load_agents_from_markdown("./batch2/", concurrent=True)

# Or limit concurrent workers for resource-constrained environments
agents = loader.load_agents_from_markdown(
    "./agents/",
    concurrent=True,
    max_file_size_mb=5.0  # Smaller files for faster processing
)
```

## Troubleshooting

### Common Issues

1. **File Not Found**: Ensure file paths are correct and files exist
2. **YAML Parsing Errors**: Check YAML frontmatter syntax in markdown files
3. **Memory Issues**: Reduce `max_file_size_mb` or process files in smaller batches
4. **Timeout Errors**: Check file sizes and network connectivity for remote files
5. **Configuration Errors**: Verify all required fields are present in agent definitions

### Debug Mode

```python
import logging
from swarms.structs import AgentLoader

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

loader = AgentLoader()

# Load with verbose output
agent = loader.load_agent_from_markdown(
    "agent.md",
    verbose=True
)
```

## Support

For questions and support:

- GitHub Issues: [https://github.com/kyegomez/swarms/issues](https://github.com/kyegomez/swarms/issues)
- Documentation: [https://docs.swarms.world](https://docs.swarms.world)
- Community: Join our Discord for real-time support