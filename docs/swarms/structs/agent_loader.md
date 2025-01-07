# Agent Loader Documentation

## Quick Reference Tables

### Configuration Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| agent_name | str | Yes | - | Unique identifier for the agent |
| system_prompt | str | Yes | - | Initial prompt that defines agent behavior |
| model_name | str | Yes | - | Model to use (e.g., "gpt-4", "claude-2") |
| max_loops | int | No | 1 | Maximum number of interaction loops |
| autosave | bool | No | True | Whether to automatically save agent state |
| dashboard | bool | No | False | Enable/disable dashboard interface |
| verbose | bool | No | True | Enable detailed logging output |
| dynamic_temperature | bool | No | True | Enable dynamic temperature adjustment |
| saved_state_path | str | No | "" | Path to save agent state file |
| user_name | str | No | "default_user" | Username for the agent |
| retry_attempts | int | No | 3 | Number of retry attempts for failed operations |
| context_length | int | No | 200000 | Maximum context length in tokens |
| return_step_meta | bool | No | False | Return metadata for each step |
| output_type | str | No | "string" | Type of output to return |
| streaming | bool | No | False | Enable streaming responses |

### Available Methods

#### AgentLoader Class

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `__init__` | `base_path: Union[str, Path]` | None | Initialize loader with base path for configurations |
| `create_agents` | `agents: List[Dict[str, Any]]`<br>`file_type: str = "csv"` | None | Create agent configurations file |
| `load_agents` | `file_type: str = "csv"` | `List[Agent]` | Load and instantiate agents from file |
| `_create_agent_csv` | `validated_agents: List[AgentConfigDict]` | None | Internal: Create CSV configuration file |
| `_create_agent_json` | `validated_agents: List[AgentConfigDict]` | None | Internal: Create JSON configuration file |
| `_load_agents_from_csv` | None | `List[Agent]` | Internal: Load agents from CSV |
| `_load_agents_from_json` | None | `List[Agent]` | Internal: Load agents from JSON |
| `_create_agent` | `validated_config: AgentConfigDict` | `Agent` | Internal: Create single agent instance |

#### ModelName Class

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `get_model_names` | None | `List[str]` | Get list of valid model names |
| `is_valid_model` | `model_name: str` | `bool` | Check if model name is valid |

#### AgentValidator Class

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `validate_config` | `config: Dict[str, Any]` | `AgentConfigDict` | Validate and convert agent configuration |

### Available Model Names

| Model Name | Enum Value |
|------------|------------|
| GPT-4 Optimized | "gpt-4o" |
| GPT-4 Optimized Mini | "gpt-4o-mini" |
| GPT-4 | "gpt-4" |
| GPT-3.5 Turbo | "gpt-3.5-turbo" |
| Claude | "claude-v1" |
| Claude 2 | "claude-2" |

## Installation

```bash
pip install swarms
```

## Basic Usage

Here's a simple example to get started:

```python
from swarms.structs.agent_loader import AgentLoader

# Initialize loader
loader = AgentLoader("agents/my_agents")

# Define single agent configuration
agent_config = {
    "agent_name": "Analysis-Agent",
    "system_prompt": "You are a data analysis expert...",
    "model_name": "gpt-4",
    "max_loops": 1,
    "autosave": True,
    "verbose": True
}

# Create and save configuration
loader.create_agents([agent_config], file_type="json")

# Load agents
agents = loader.load_agents(file_type="json")
```

## Detailed Components

### ModelName Enum

The `ModelName` enum defines valid model names and provides utility methods:

```python
from enum import Enum

class ModelName(str, Enum):
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
    GPT4 = "gpt-4"
    GPT35_TURBO = "gpt-3.5-turbo"
    CLAUDE = "claude-v1"
    CLAUDE2 = "claude-2"

    @classmethod
    def get_model_names(cls) -> List[str]:
        return [model.value for model in cls]

    @classmethod
    def is_valid_model(cls, model_name: str) -> bool:
        return model_name in cls.get_model_names()
```

### AgentValidator

The validator ensures configuration integrity:

```python
class AgentValidator:
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> AgentConfigDict:
        # Validate model name
        model_name = str(config["model_name"])
        if not ModelName.is_valid_model(model_name):
            raise AgentValidationError(
                f"Invalid model name. Must be one of: {ModelName.get_model_names()}",
                "model_name",
                model_name
            )
        
        # Convert and validate other fields
        validated_config: AgentConfigDict = {
            "agent_name": str(config.get("agent_name", "")),
            "system_prompt": str(config.get("system_prompt", "")),
            # ... other fields
        }
        
        return validated_config
```

## Advanced Usage Examples

### Multiple Agents with Different Models

```python
agent_configs = [
    {
        "agent_name": "Research-Agent",
        "system_prompt": "You are a research assistant...",
        "model_name": "gpt-4",
        "max_loops": 2,
        "autosave": True
    },
    {
        "agent_name": "Writing-Agent",
        "system_prompt": "You are a professional writer...",
        "model_name": "claude-2",
        "max_loops": 1,
        "autosave": True
    }
]

loader = AgentLoader("agents/multi_agents")
loader.create_agents(agent_configs, file_type="json")
```

### Custom File Paths and Error Handling

```python
from pathlib import Path

try:
    loader = AgentLoader(Path("custom/path/agents"))
    agents = loader.load_agents(file_type="csv")
except FileNotFoundError:
    print("Configuration file not found")
except AgentValidationError as e:
    print(f"Invalid configuration: {e}")
```

## Error Handling

The system uses custom exceptions for clear error handling:

```python
@dataclass
class AgentValidationError(Exception):
    message: str
    field: str
    value: Any

    def __str__(self) -> str:
        return f"Validation error in field '{self.field}': {self.message}. Got value: {self.value}"
```

Common error scenarios:

1. Invalid model name:
```python
try:
    config = {"model_name": "invalid-model"}
    AgentValidator.validate_config(config)
except AgentValidationError as e:
    print(f"Error: {e}")  # Will show available model names
```

2. Missing required fields:
```python
try:
    config = {}  # Missing required fields
    AgentValidator.validate_config(config)
except KeyError as e:
    print(f"Missing required field: {e}")
```

## Best Practices

### 1. Configuration Management

```python
# Use consistent paths
from pathlib import Path
base_path = Path("configurations/agents")
loader = AgentLoader(base_path)

# Group related agents
research_agents = {
    "agent_name": "Research-Agent",
    "system_prompt": "You are a research assistant...",
    "model_name": "gpt-4"
}
```

### 2. Validation

```python
# Pre-validate configurations
from swarms.structs.agent_loader import AgentValidator

def prepare_agent_config(config: Dict[str, Any]) -> AgentConfigDict:
    try:
        return AgentValidator.validate_config(config)
    except AgentValidationError as e:
        logging.error(f"Configuration error: {e}")
        raise
```

### 3. File Type Selection

- Use CSV for:
  - Human-readable configurations
  - Simple data structures
  - Manual editing

- Use JSON for:
  - Complex nested structures
  - Programmatic access
  - Better type preservation

### 4. Error Recovery

```python
import logging

def safe_load_agents(loader: AgentLoader, file_type: str = "json") -> List[Agent]:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return loader.load_agents(file_type=file_type)
        except FileNotFoundError:
            logging.error(f"Attempt {attempt + 1}: File not found")
            continue
        except AgentValidationError as e:
            logging.error(f"Attempt {attempt + 1}: Validation error - {e}")
            continue
    raise Exception(f"Failed to load agents after {max_retries} attempts")
```

## Performance Considerations

1. File Size:
   - CSV files are typically smaller than JSON
   - Consider compression for large configurations

2. Loading Time:
   - JSON parsing is generally faster than CSV
   - Batch operations are more efficient than individual loads

3. Memory Usage:
   - Load agents only when needed
   - Consider implementing pagination for large sets of agents

## Security Considerations

1. File Permissions:
   - Set appropriate file permissions for configuration files
   - Use secure directories for saved states

2. Input Validation:
   - Always validate user input before creating configurations
   - Sanitize file paths to prevent directory traversal

3. Sensitive Data:
   - Don't store sensitive information in configuration files
   - Use environment variables for sensitive data

## Debugging Tips

1. Enable verbose mode:
```python
agent_config = {
    "verbose": True,
    # other configurations...
}
```

2. Use logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

3. Validate configurations independently:
```python
validated_config = AgentValidator.validate_config(agent_config)
print(json.dumps(validated_config, indent=2))
```

## Maintenance

1. Regular validation:
```python
def validate_all_configs(directory: Path):
    loader = AgentLoader(directory)
    try:
        agents = loader.load_agents()
        print(f"All {len(agents)} configurations are valid")
    except Exception as e:
        print(f"Validation failed: {e}")
```

2. Backup configurations:
```python
from shutil import copy2
from datetime import datetime

def backup_configs(loader: AgentLoader):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = loader.base_path.with_suffix(f".backup_{timestamp}")
    copy2(loader.base_path, backup_path)
```

This documentation provides a comprehensive guide to using the Agent Loader system effectively. For specific use cases or additional examples, please refer to the relevant sections above.