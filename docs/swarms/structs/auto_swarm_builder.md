# AutoSwarmBuilder Documentation

The `AutoSwarmBuilder` is a powerful class that automatically builds and manages swarms of AI agents to accomplish complex tasks. It uses a sophisticated boss agent system to delegate work and create specialized agents as needed.

## Overview

The AutoSwarmBuilder is designed to:

| Feature | Description |
|---------|-------------|
| Automatic Agent Creation | Automatically create and coordinate multiple AI agents |
| Task Delegation | Delegate tasks to specialized agents based on task requirements |
| Agent Communication Management | Manage communication between agents through a swarm router |
| Complex Workflow Handling | Handle complex workflows with various execution types |
| Architecture Pattern Support | Support different multi-agent architecture patterns |
| Error Handling & Logging | Provide comprehensive error handling and logging |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| name | str | "auto-swarm-builder" | The name of the swarm |
| description | str | "Auto Swarm Builder" | A description of the swarm's purpose |
| verbose | bool | True | Whether to output detailed logs |
| max_loops | int | 1 | Maximum number of execution loops |
| model_name | str | "gpt-4.1" | The LLM model to use for the boss agent |
| generate_router_config | bool | False | Whether to generate router configuration |
| interactive | bool | False | Whether to enable interactive mode |
| max_tokens | int | 8000 | Maximum tokens for the LLM responses |
| execution_type | str | "return-agents" | Type of execution to perform |

## Execution Types

The `execution_type` parameter controls how the AutoSwarmBuilder operates:

| Execution Type                  | Description                                               |
|----------------------------------|-----------------------------------------------------------|
| **"return-agents"**              | Creates and returns a list of Agent objects (default)     |
| **"execute-swarm-router"**       | Executes the swarm router with the created agents         |
| **"return-swarm-router-config"** | Returns the swarm router configuration as a dictionary    |
| **"return-agent-configurations"**| Returns agent configurations as a dictionary              |

## Core Methods

### run(task: str, *args, **kwargs)

Executes the swarm on a given task based on the configured execution type.

**Parameters:**

- `task` (str): The task to execute

- `*args`: Additional positional arguments

- `**kwargs`: Additional keyword arguments

**Returns:**

- The result of the swarm execution (varies by execution_type)

**Raises:**

- `Exception`: If there's an error during execution

### create_agents(task: str)

Creates specialized agents for a given task using the boss agent system.

**Parameters:**

- `task` (str): The task to create agents for

**Returns:**

- `List[Agent]`: List of created agents

**Raises:**

- `Exception`: If there's an error during agent creation

### build_agent(agent_name: str, agent_description: str, agent_system_prompt: str)

Builds a single agent with specified parameters and enhanced error handling.

**Parameters:**

| Parameter             | Type  | Description                    |
|-----------------------|-------|--------------------------------|
| `agent_name`          | str   | Name of the agent              |
| `agent_description`   | str   | Description of the agent       |
| `agent_system_prompt` | str   | System prompt for the agent    |

**Returns:**

- `Agent`: The constructed agent

**Raises:**

- `Exception`: If there's an error during agent construction

### create_router_config(task: str)

Creates a swarm router configuration for a given task.

**Parameters:**

- `task` (str): The task to create router config for

**Returns:**

- `dict`: Swarm router configuration dictionary

**Raises:**

- `Exception`: If there's an error creating the configuration

### initialize_swarm_router(agents: List[Agent], task: str)

Initializes and runs the swarm router with the provided agents.

**Parameters:**

- `agents` (List[Agent]): List of agents to use

- `task` (str): The task to execute

**Returns:**

- `Any`: The result of the swarm router execution

**Raises:**

- `Exception`: If there's an error during router initialization or execution

### batch_run(tasks: List[str])

Executes the swarm on multiple tasks sequentially.

**Parameters:**

- `tasks` (List[str]): List of tasks to execute

**Returns:**

- `List[Any]`: List of results from each task execution

**Raises:**

- `Exception`: If there's an error during batch execution

### list_types()

Returns the available execution types.

**Returns:**

- `List[str]`: List of available execution types

### dict_to_agent(output: dict)

Converts a dictionary output to a list of Agent objects.

**Parameters:**

- `output` (dict): Dictionary containing agent configurations

**Returns:**

- `List[Agent]`: List of constructed agents

### build_llm_agent(config: BaseModel)

Builds an LLM agent for configuration generation.

**Parameters:**

- `config` (BaseModel): Pydantic model for response format

**Returns:**

- `LiteLLM`: Configured LLM agent

### reliability_check()

Performs reliability checks on the AutoSwarmBuilder configuration.

**Raises:**

- `ValueError`: If max_loops is set to 0

## Configuration Classes

### AgentConfig

Configuration model for individual agents in a swarm.

**Fields:**

| Field           | Type    | Description                                                                                   |
|-----------------|---------|-----------------------------------------------------------------------------------------------|
| `name`          | str     | Unique identifier for the agent                                                               |
| `description`   | str     | Comprehensive description of the agent's purpose and capabilities                             |
| `system_prompt` | str     | Detailed system prompt defining agent behavior                                                |
| `goal`          | str     | Primary objective the agent is tasked with achieving                                          |
| `model_name`    | str     | LLM model to use for the agent (e.g., 'gpt-4o-mini')                                         |
| `temperature`   | float   | Controls randomness of responses (0.0-1.0)                                                    |
| `max_loops`     | int     | Maximum number of execution loops (typically 1)                                               |

### AgentsConfig

Configuration model for a list of agents in a swarm.

**Fields:**

- `agents` (List[AgentConfig]): List of agent configurations

### SwarmRouterConfig

Configuration model for SwarmRouter.

**Fields:**

- `name` (str): Name of the team of agents
- `description` (str): Description of the team of agents
- `agents` (List[AgentConfig]): List of agent configurations
- `swarm_type` (SwarmType): Type of multi-agent structure to use
- `rearrange_flow` (Optional[str]): Flow configuration for AgentRearrange structure
- `rules` (Optional[str]): Rules to inject into every agent's system prompt
- `task` (str): The task to be executed by the swarm

## Multi-Agent Architecture Types

The AutoSwarmBuilder supports various multi-agent architecture patterns:

| Architecture Type         | Description                                               |
|--------------------------|-----------------------------------------------------------|
| **AgentRearrange**       | Dynamic task reallocation based on agent performance      |
| **MixtureOfAgents**      | Parallel processing with specialized agents               |
| **SpreadSheetSwarm**     | Structured data processing with coordinated workflows     |
| **SequentialWorkflow**   | Linear task progression with handoffs                     |
| **ConcurrentWorkflow**   | Parallel execution with coordination                      |
| **GroupChat**            | Collaborative discussion and consensus-building           |
| **MultiAgentRouter**     | Intelligent routing and load balancing                    |
| **AutoSwarmBuilder**     | Self-organizing and self-optimizing teams                 |
| **HiearchicalSwarm**     | Layered decision-making with management tiers             |
| **MajorityVoting**       | Democratic decision-making with voting                    |
| **MALT**                 | Multi-agent learning and training                         |
| **CouncilAsAJudge**      | Deliberative decision-making with expert panels           |
| **InteractiveGroupChat** | Dynamic group interactions                                |
| **HeavySwarm**           | High-capacity processing with multiple agents             |

## Examples

### Example 1: Basic Content Creation Swarm

```python
from swarms.structs.auto_swarm_builder import AutoSwarmBuilder

# Initialize the swarm builder with default settings
swarm = AutoSwarmBuilder(
    name="Content Creation Swarm",
    description="A swarm specialized in creating high-quality content"
)

# Run the swarm on a content creation task
result = swarm.run(
    "Create a comprehensive blog post about artificial intelligence in healthcare, "
    "including current applications, future trends, and ethical considerations."
)
```

### Example 2: Advanced Configuration with Custom Model

```python
from swarms.structs.auto_swarm_builder import AutoSwarmBuilder

# Initialize with custom configuration
swarm = AutoSwarmBuilder(
    name="Data Analysis Swarm",
    description="A swarm specialized in data analysis and visualization",
    model_name="gpt-4o",
    max_tokens=12000,
    verbose=True,
    execution_type="return-agents"
)

# Run the swarm on a data analysis task
result = swarm.run(
    "Analyze the provided sales data and create a detailed report with visualizations "
    "showing trends, patterns, and recommendations for improvement."
)
```

### Example 3: Getting Agent Configurations Only

```python
from swarms.structs.auto_swarm_builder import AutoSwarmBuilder

# Initialize to return agent configurations
swarm = AutoSwarmBuilder(
    name="Marketing Swarm",
    description="A swarm for marketing strategy development",
    execution_type="return-agent-configurations"
)

# Get agent configurations without executing
agent_configs = swarm.run(
    "Create a comprehensive marketing strategy for a new tech product launch"
)

print("Generated agents:")
for agent in agent_configs["agents"]:
    print(f"- {agent['name']}: {agent['description']}")
```

### Example 4: Getting Swarm Router Configuration

```python
from swarms.structs.auto_swarm_builder import AutoSwarmBuilder

# Initialize to return router configuration
swarm = AutoSwarmBuilder(
    name="Research Swarm",
    description="A swarm for research and analysis",
    execution_type="return-swarm-router-config"
)

# Get the complete swarm router configuration
router_config = swarm.run(
    "Conduct market research on renewable energy trends and create a detailed report"
)

print(f"Swarm Type: {router_config['swarm_type']}")
print(f"Number of Agents: {len(router_config['agents'])}")
```

### Example 5: Batch Processing Multiple Tasks

```python
from swarms.structs.auto_swarm_builder import AutoSwarmBuilder

# Initialize the swarm builder
swarm = AutoSwarmBuilder(
    name="Multi-Task Swarm",
    description="A swarm capable of handling multiple diverse tasks",
    max_loops=2,
    interactive=True
)

# Define multiple tasks
tasks = [
    "Create a marketing strategy for a new product launch",
    "Analyze customer feedback and generate improvement suggestions",
    "Develop a project timeline for the next quarter"
]

# Run the swarm on all tasks
results = swarm.batch_run(tasks)

# Process results
for i, result in enumerate(results):
    print(f"Task {i+1} completed: {result}")
```

### Example 6: Interactive Mode with Custom Parameters

```python
from swarms.structs.auto_swarm_builder import AutoSwarmBuilder

# Initialize with interactive mode and custom settings
swarm = AutoSwarmBuilder(
    name="Interactive Swarm",
    description="An interactive swarm for real-time collaboration",
    model_name="claude-3-sonnet-20240229",
    max_tokens=16000,
    interactive=True,
    generate_router_config=True,
    verbose=True
)

# Run with interactive capabilities
result = swarm.run(
    "Help me design a user interface for a mobile app that helps people track their fitness goals"
)
```

## Best Practices

!!! tip "Task Definition"
    - Provide clear, specific task descriptions
    - Include any relevant context or constraints
    - Specify expected output format if needed
    - Break complex tasks into smaller, manageable components

!!! note "Configuration"
    - Set appropriate `max_loops` based on task complexity
    - Use `verbose=True` during development for debugging
    - Choose the right `execution_type` for your use case:
        - Use `"return-agents"` for direct agent interaction
        - Use `"return-agent-configurations"` for inspection
        - Use `"return-swarm-router-config"` for configuration analysis
    - Set `max_tokens` appropriately based on expected response length
    - Use `interactive=True` for real-time collaboration scenarios

!!! note "Model Selection"
    - Choose appropriate `model_name` based on task requirements
    - Consider model capabilities and token limits
    - Use more powerful models for complex reasoning tasks
    - Balance performance with cost considerations

!!! warning "Error Handling"
    - The class includes comprehensive error handling
    - All methods include try-catch blocks with detailed logging
    - Errors are propagated with full stack traces for debugging
    - Always handle exceptions in your calling code

!!! info "Performance Optimization"
    - Use `batch_run()` for processing multiple similar tasks
    - Consider using `generate_router_config=True` for complex workflows
    - Monitor token usage with `max_tokens` parameter
    - Use appropriate `swarm_type` for your specific use case

## Notes

!!! info "Architecture"
    - The AutoSwarmBuilder uses a sophisticated boss agent system with comprehensive system prompts
    - Agents are created dynamically based on task requirements using AI-powered analysis
    - The system supports multiple execution types for different use cases
    - Built-in logging and error handling with detailed traceback information
    - Results are returned in structured formats (agents, configurations, or execution results)
    - Supports various multi-agent architecture patterns through SwarmRouter
    - Uses LiteLLM for flexible model support and response formatting

!!! info "Dependencies"
    - Requires `loguru` for logging
    - Uses `pydantic` for data validation and configuration
    - Integrates with `swarms.structs.agent.Agent` for individual agents
    - Uses `swarms.structs.swarm_router.SwarmRouter` for coordination
    - Leverages `swarms.utils.litellm_wrapper.LiteLLM` for LLM interactions

!!! info "System Prompt"
    - The boss agent uses a comprehensive system prompt that includes:
        - Multi-agent architecture design principles
        - Agent creation guidelines and best practices
        - Support for various swarm types and patterns
        - Quality assurance and governance frameworks
        - Error handling and recovery procedures
