# Board of Directors Configuration

The Board of Directors feature in Swarms provides a sophisticated configuration system that allows you to enable, customize, and manage the collective decision-making capabilities of the framework.

## Overview

The Board of Directors configuration system provides:

- **Feature Control**: Enable or disable the Board of Directors feature globally
- **Board Composition**: Configure default board sizes and member roles
- **Operational Settings**: Set decision thresholds, voting mechanisms, and consensus parameters
- **Template Management**: Predefined board templates for common use cases
- **Environment Integration**: Configuration through environment variables and files

## Configuration Management

### `BoardConfig` Class

The `BoardConfig` class manages all configuration for the Board of Directors feature:

```python
from swarms.config.board_config import BoardConfig

# Create configuration with custom settings
config = BoardConfig(
    config_file_path="board_config.json",
    config_data={
        "board_feature_enabled": True,
        "default_board_size": 5,
        "decision_threshold": 0.7
    }
)
```

### Configuration Sources

The configuration system loads settings from multiple sources in priority order:

1. **Environment Variables** (highest priority)
2. **Configuration File**
3. **Explicit Config Data**
4. **Default Values** (lowest priority)

## Environment Variables

You can configure the Board of Directors feature using environment variables:

```bash
# Enable the Board of Directors feature
export SWARMS_BOARD_FEATURE_ENABLED=true

# Set default board size
export SWARMS_DEFAULT_BOARD_SIZE=5

# Configure decision threshold
export SWARMS_DECISION_THRESHOLD=0.7

# Enable voting mechanisms
export SWARMS_ENABLE_VOTING=true

# Enable consensus building
export SWARMS_ENABLE_CONSENSUS=true

# Set default board model
export SWARMS_DEFAULT_BOARD_MODEL=gpt-4o

# Enable verbose logging
export SWARMS_VERBOSE_LOGGING=true

# Set maximum board meeting duration
export SWARMS_MAX_BOARD_MEETING_DURATION=300

# Enable auto fallback to Director mode
export SWARMS_AUTO_FALLBACK_TO_DIRECTOR=true
```

## Configuration File

Create a JSON configuration file for persistent settings:

```json
{
    "board_feature_enabled": true,
    "default_board_size": 5,
    "decision_threshold": 0.7,
    "enable_voting": true,
    "enable_consensus": true,
    "default_board_model": "gpt-4o",
    "verbose_logging": true,
    "max_board_meeting_duration": 300,
    "auto_fallback_to_director": true,
    "custom_board_templates": {
        "financial": {
            "roles": [
                {"name": "CFO", "weight": 1.5, "expertise": ["finance", "risk_management"]},
                {"name": "Investment_Advisor", "weight": 1.3, "expertise": ["investments", "analysis"]}
            ]
        }
    }
}
```

## Configuration Functions

### Feature Control

```python
from swarms.config.board_config import (
    enable_board_feature,
    disable_board_feature,
    is_board_feature_enabled
)

# Check if feature is enabled
if not is_board_feature_enabled():
    # Enable the feature
    enable_board_feature()
    print("Board of Directors feature enabled")

# Disable the feature
disable_board_feature()
```

### Board Composition

```python
from swarms.config.board_config import (
    set_board_size,
    get_board_size
)

# Set default board size
set_board_size(7)

# Get current board size
current_size = get_board_size()
print(f"Default board size: {current_size}")
```

### Decision Settings

```python
from swarms.config.board_config import (
    set_decision_threshold,
    get_decision_threshold,
    enable_voting,
    disable_voting,
    enable_consensus,
    disable_consensus
)

# Set decision threshold (0.0 to 1.0)
set_decision_threshold(0.75)  # 75% majority required

# Get current threshold
threshold = get_decision_threshold()
print(f"Decision threshold: {threshold}")

# Enable/disable voting mechanisms
enable_voting()
disable_voting()

# Enable/disable consensus building
enable_consensus()
disable_consensus()
```

### Model Configuration

```python
from swarms.config.board_config import (
    set_board_model,
    get_board_model
)

# Set default model for board members
set_board_model("gpt-4o")

# Get current model
model = get_board_model()
print(f"Default board model: {model}")
```

### Logging Configuration

```python
from swarms.config.board_config import (
    enable_verbose_logging,
    disable_verbose_logging,
    is_verbose_logging_enabled
)

# Enable verbose logging
enable_verbose_logging()

# Check logging status
if is_verbose_logging_enabled():
    print("Verbose logging is enabled")

# Disable verbose logging
disable_verbose_logging()
```

### Meeting Duration

```python
from swarms.config.board_config import (
    set_max_board_meeting_duration,
    get_max_board_meeting_duration
)

# Set maximum meeting duration in seconds
set_max_board_meeting_duration(600)  # 10 minutes

# Get current duration
duration = get_max_board_meeting_duration()
print(f"Max meeting duration: {duration} seconds")
```

### Fallback Configuration

```python
from swarms.config.board_config import (
    enable_auto_fallback_to_director,
    disable_auto_fallback_to_director,
    is_auto_fallback_enabled
)

# Enable automatic fallback to Director mode
enable_auto_fallback_to_director()

# Check fallback status
if is_auto_fallback_enabled():
    print("Auto fallback to Director mode is enabled")

# Disable fallback
disable_auto_fallback_to_director()
```

## Board Templates

### Default Templates

The configuration system provides predefined board templates for common use cases:

```python
from swarms.config.board_config import get_default_board_template

# Get standard board template
standard_template = get_default_board_template("standard")
print("Standard template roles:", standard_template["roles"])

# Get executive board template
executive_template = get_default_board_template("executive")
print("Executive template roles:", executive_template["roles"])

# Get advisory board template
advisory_template = get_default_board_template("advisory")
print("Advisory template roles:", advisory_template["roles"])
```

### Template Structure

Each template defines the board composition:

```python
# Standard template structure
standard_template = {
    "roles": [
        {
            "name": "Chairman",
            "weight": 1.5,
            "expertise": ["leadership", "strategy"]
        },
        {
            "name": "Vice-Chairman", 
            "weight": 1.2,
            "expertise": ["operations", "coordination"]
        },
        {
            "name": "Secretary",
            "weight": 1.0,
            "expertise": ["documentation", "communication"]
        }
    ]
}
```

### Custom Templates

Create custom board templates for specific use cases:

```python
from swarms.config.board_config import (
    add_custom_board_template,
    get_custom_board_template,
    list_custom_templates
)

# Define a custom financial analysis board
financial_template = {
    "roles": [
        {
            "name": "CFO",
            "weight": 1.5,
            "expertise": ["finance", "risk_management", "budgeting"]
        },
        {
            "name": "Investment_Advisor",
            "weight": 1.3,
            "expertise": ["investments", "market_analysis", "portfolio_management"]
        },
        {
            "name": "Compliance_Officer",
            "weight": 1.2,
            "expertise": ["compliance", "regulations", "legal"]
        }
    ]
}

# Add custom template
add_custom_board_template("financial_analysis", financial_template)

# Get custom template
template = get_custom_board_template("financial_analysis")

# List all custom templates
templates = list_custom_templates()
print("Available custom templates:", templates)
```

## Configuration Validation

The configuration system includes comprehensive validation:

```python
from swarms.config.board_config import validate_configuration

# Validate current configuration
try:
    validation_result = validate_configuration()
    print("Configuration is valid:", validation_result.is_valid)
    if not validation_result.is_valid:
        print("Validation errors:", validation_result.errors)
except Exception as e:
    print(f"Configuration validation failed: {e}")
```

## Configuration Persistence

### Save Configuration

```python
from swarms.config.board_config import save_configuration

# Save current configuration to file
save_configuration("my_board_config.json")
```

### Load Configuration

```python
from swarms.config.board_config import load_configuration

# Load configuration from file
config = load_configuration("my_board_config.json")
```

### Reset to Defaults

```python
from swarms.config.board_config import reset_to_defaults

# Reset all configuration to default values
reset_to_defaults()
```

## Integration with BoardOfDirectorsSwarm

The configuration system integrates seamlessly with the BoardOfDirectorsSwarm:

```python
from swarms.structs.board_of_directors_swarm import BoardOfDirectorsSwarm
from swarms.config.board_config import (
    enable_board_feature,
    set_decision_threshold,
    get_default_board_template
)

# Enable the feature globally
enable_board_feature()

# Set global decision threshold
set_decision_threshold(0.7)

# Get a board template
template = get_default_board_template("executive")

# Create board members from template
board_members = []
for role_config in template["roles"]:
    agent = Agent(
        agent_name=role_config["name"],
        agent_description=f"Board member with expertise in {', '.join(role_config['expertise'])}",
        model_name="gpt-4o-mini"
    )
    board_member = BoardMember(
        agent=agent,
        role=BoardMemberRole.EXECUTIVE_DIRECTOR,
        voting_weight=role_config["weight"],
        expertise_areas=role_config["expertise"]
    )
    board_members.append(board_member)

# Create the swarm with configured settings
board_swarm = BoardOfDirectorsSwarm(
    board_members=board_members,
    agents=worker_agents,
    decision_threshold=0.7,  # Uses global setting
    enable_voting=True,
    enable_consensus=True
)
```

## Best Practices

1. **Environment Variables**: Use environment variables for deployment-specific settings
2. **Configuration Files**: Use JSON files for persistent, version-controlled settings
3. **Validation**: Always validate configuration before deployment
4. **Templates**: Use predefined templates for common use cases
5. **Customization**: Create custom templates for domain-specific requirements
6. **Monitoring**: Enable verbose logging for debugging and monitoring
7. **Fallback**: Configure fallback mechanisms for reliability

## Error Handling

The configuration system includes comprehensive error handling:

```python
from swarms.config.board_config import BoardConfig

try:
    config = BoardConfig(
        config_file_path="invalid_config.json"
    )
except Exception as e:
    print(f"Configuration loading failed: {e}")
    # Handle error appropriately
```

## Performance Considerations

- **Caching**: Configuration values are cached for improved performance
- **Lazy Loading**: Templates are loaded on-demand
- **Validation**: Configuration validation is performed efficiently
- **Memory Management**: Configuration objects are lightweight and efficient

---

For more information on using the Board of Directors feature, see the [BoardOfDirectorsSwarm Documentation](https://docs.swarms.world/en/latest/swarms/structs/board_of_directors_swarm/). 