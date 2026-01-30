# AutoSwarmBuilder Documentation

The `AutoSwarmBuilder` is a powerful class that automatically builds and manages swarms of AI agents to accomplish complex tasks. It uses a sophisticated boss agent system with comprehensive design principles to delegate work and create specialized agents as needed.

The AutoSwarmBuilder is designed to:

| Feature | Description |
|---------|-------------|
| **Automatic Agent Creation** | Automatically create and coordinate multiple AI agents with distinct personalities and capabilities |
| **Intelligent Task Delegation** | Delegate tasks to specialized agents based on comprehensive task analysis and requirements |
| **Advanced Agent Communication** | Manage sophisticated communication protocols between agents through a swarm router |
| **Multiple Execution Types** | Support 6 different execution types for various use cases and workflows |
| **Comprehensive Architecture Support** | Support 13+ different multi-agent architecture patterns and coordination strategies |
| **Robust Error Handling** | Provide comprehensive error handling, logging, and recovery procedures |
| **Dynamic Agent Specification** | Create agents with detailed specifications including roles, personalities, and capabilities |
| **Flexible Configuration** | Support extensive configuration options for models, tokens, temperature, and behavior |
| **Batch Processing** | Handle multiple tasks efficiently with batch processing capabilities |
| **Interactive Mode** | Support real-time interactive collaboration and decision-making |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | "auto-swarm-builder" | The name of the swarm |
| `description` | str | "Auto Swarm Builder" | A description of the swarm's purpose |
| `verbose` | bool | True | Whether to output detailed logs |
| `max_loops` | int | 1 | Maximum number of execution loops |
| `model_name` | str | "gpt-4.1" | The LLM model to use for the boss agent |
| `generate_router_config` | bool | False | Whether to generate router configuration |
| `interactive` | bool | False | Whether to enable interactive mode |
| `max_tokens` | int | 8000 | Maximum tokens for the LLM responses |
| `execution_type` | str | "return-agents" | Type of execution to perform (see Execution Types) |
| `system_prompt` | str | BOSS_SYSTEM_PROMPT | System prompt for the boss agent |
| `additional_llm_args` | dict | {} | Additional arguments to pass to the LLM |

## Execution Types

The `execution_type` parameter controls how the AutoSwarmBuilder operates:

| Execution Type                  | Description                                               |
|----------------------------------|-----------------------------------------------------------|
| **"return-agents"**              | Creates and returns agent specifications as a dictionary (default) |
| **"return-swarm-router-config"** | Returns the swarm router configuration as a dictionary    |
| **"return-agents-objects"**     | Returns agent objects created from specifications         |

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

### create_agents_from_specs(agents_dictionary: Any)

Create agents from agent specifications.

**Parameters:**

- `agents_dictionary`: Dictionary containing agent specifications

**Returns:**

- `List[Agent]`: List of created agents

### dict_to_agent(output: dict)

Converts a dictionary output to a list of Agent objects.

**Parameters:**

- `output` (dict): Dictionary containing agent configurations

**Returns:**

- `List[Agent]`: List of constructed agents

### _execute_task(task: str)

Execute a task by creating agents and initializing the swarm router.

**Parameters:**

- `task` (str): The task to execute

**Returns:**

- `Any`: The result of the swarm router execution

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

### AgentSpec

Configuration for an individual agent specification with comprehensive options.

**Fields:**

| Field                | Type    | Description                                                                                   |
|----------------------|---------|-----------------------------------------------------------------------------------------------|
| `agent_name`         | str     | Unique name assigned to the agent, identifying its role and functionality                    |
| `description`        | str     | Detailed explanation of the agent's purpose, capabilities, and specific tasks                |
| `system_prompt`      | str     | Initial instruction or context provided to guide agent behavior and responses               |
| `model_name`         | str     | AI model name for processing tasks (e.g., 'gpt-4o', 'gpt-4o-mini', 'openai/o3-mini')      |
| `auto_generate_prompt`| bool   | Flag indicating whether the agent should automatically create prompts                        |
| `max_tokens`         | int     | Maximum number of tokens allowed in agent responses                                          |
| `temperature`        | float   | Parameter controlling randomness of agent output (lower = more deterministic)               |
| `role`               | str     | Designated role within the swarm influencing behavior and interactions                       |
| `max_loops`          | int     | Maximum number of times the agent can repeat its task for iterative processing              |
| `goal`               | str     | The primary objective or desired outcome the agent is tasked with achieving                  |

### Agents

Configuration for a collection of agents that work together as a swarm.

**Fields:**

- `agents` (List[AgentSpec]): List containing specifications of each agent participating in the swarm

### AgentConfig

Configuration model for individual agents in a swarm.

**Fields:**

| Field           | Type    | Description                                                                                   |
|-----------------|---------|-----------------------------------------------------------------------------------------------|
| `agent_name`    | str     | Unique identifier for the agent                                                               |
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
| **CouncilAsAJudge**      | Deliberative decision-making with expert panels           |
| **HeavySwarm**           | High-capacity processing with multiple agents             |

## Boss System Prompt

The AutoSwarmBuilder uses a comprehensive `BOSS_SYSTEM_PROMPT` that embodies sophisticated multi-agent architecture design principles. This system prompt guides the boss agent in creating highly effective agent teams.

### Core Design Principles

The boss system prompt includes six fundamental design principles:

1. **Comprehensive Task Analysis**
   - Thoroughly deconstruct tasks into fundamental components and sub-tasks
   - Identify specific skills, knowledge domains, and personality traits required
   - Analyze challenges, dependencies, and coordination requirements
   - Map optimal workflows, information flow patterns, and decision-making hierarchies

2. **Agent Design Excellence**
   - Create agents with crystal-clear, specific purposes and domain expertise
   - Design distinct, complementary personalities that enhance team dynamics
   - Ensure agents are self-aware of limitations and know when to seek assistance
   - Create agents that effectively communicate progress, challenges, and insights

3. **Comprehensive Agent Framework**
   - **Role & Purpose**: Precise description of responsibilities and authority
   - **Personality Profile**: Distinct characteristics influencing thinking patterns
   - **Expertise Matrix**: Specific knowledge domains, skill sets, and capabilities
   - **Communication Protocol**: How agents present information and interact
   - **Decision-Making Framework**: Systematic approach to problem-solving
   - **Limitations & Boundaries**: Clear constraints and operational boundaries
   - **Collaboration Strategy**: How agents work together and share knowledge

4. **Advanced System Prompt Engineering**
   - Detailed role and purpose explanations with context and scope
   - Rich personality descriptions with behavioral guidelines
   - Comprehensive capabilities, tools, and resource specifications
   - Detailed communication protocols and reporting requirements
   - Systematic problem-solving approaches with decision-making frameworks
   - Collaboration guidelines and conflict resolution procedures
   - Quality standards, success criteria, and performance metrics
   - Error handling, recovery procedures, and escalation protocols

5. **Multi-Agent Coordination Architecture**
   - Design robust communication channels and protocols between agents
   - Establish clear task handoff procedures and information sharing mechanisms
   - Create feedback loops for continuous improvement and adaptation
   - Implement comprehensive error handling and recovery procedures
   - Define escalation paths for complex issues and decision-making hierarchies

6. **Quality Assurance & Governance**
   - Set measurable success criteria for each agent and the overall system
   - Implement verification steps, validation procedures, and quality checks
   - Create mechanisms for self-assessment, peer review, and continuous improvement
   - Establish protocols for handling edge cases and unexpected situations
   - Design governance structures for oversight, accountability, and performance management

### Output Requirements

The boss system prompt ensures that when creating multi-agent systems, the following are provided:

1. **Agent Specifications**: Comprehensive role statements, personality profiles, capabilities, limitations, communication styles, and collaboration strategies
2. **System Prompts**: Complete, detailed prompts embodying each agent's identity and capabilities
3. **Architecture Design**: Team structure, communication flow patterns, task distribution strategies, quality control measures, and error handling procedures

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
    model_name="gpt-4.1",
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
    execution_type="return-agents"
)

# Get agent configurations without executing
agent_configs = swarm.run(
    "Create a comprehensive marketing strategy for a new tech product launch"
)

print("Generated agents:")
for agent in agent_configs["agents"]:
    print(f"- {agent['agent_name']}: {agent['description']}")
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

### Example 7: Getting Agent Objects

```python
from swarms.structs.auto_swarm_builder import AutoSwarmBuilder

# Initialize to return agent objects
swarm = AutoSwarmBuilder(
    name="Specification Swarm",
    description="A swarm for generating agent specifications",
    execution_type="return-agents-objects"
)

# Get agent objects
agents = swarm.run(
    "Create a team of agents for analyzing customer feedback and generating actionable insights"
)

print(f"Created {len(agents)} agents:")
for agent in agents:
    print(f"- {agent.agent_name}: {agent.description}")
```

### Example 8: Getting Agent Dictionary

```python
from swarms.structs.auto_swarm_builder import AutoSwarmBuilder

# Initialize to return agent dictionary
swarm = AutoSwarmBuilder(
    name="Dictionary Swarm",
    description="A swarm for generating agent dictionaries",
    execution_type="return-agents"
)

# Get agent configurations as dictionary
agent_dict = swarm.run(
    "Create a marketing team to develop a comprehensive social media strategy"
)

print("Agent Dictionary:")
for agent in agent_dict["agents"]:
    print(f"- {agent['agent_name']}: {agent['description']}")
    print(f"  Model: {agent['model_name']}")
    print(f"  Role: {agent['role']}")
    print(f"  Temperature: {agent['temperature']}")
```

### Example 9: Custom System Prompt

```python
from swarms.structs.auto_swarm_builder import AutoSwarmBuilder

# Custom system prompt for specialized domain
custom_prompt = """
You are an expert in financial analysis and risk assessment. 
Create specialized agents for financial modeling, risk analysis, 
and investment strategy development. Focus on quantitative analysis, 
regulatory compliance, and market research capabilities.
"""

# Initialize with custom system prompt
swarm = AutoSwarmBuilder(
    name="Financial Analysis Swarm",
    description="A specialized swarm for financial analysis",
    system_prompt=custom_prompt,
    model_name="gpt-4.1",
    max_tokens=12000
)

# Run with custom prompt
result = swarm.run(
    "Analyze the financial health of a tech startup and provide investment recommendations"
)
```

### Example 10: Advanced Agent Configuration

```python
from swarms.structs.auto_swarm_builder import AutoSwarmBuilder

# Initialize with advanced configuration
swarm = AutoSwarmBuilder(
    name="Advanced Swarm",
    description="A highly configured swarm with advanced settings",
    model_name="gpt-4.1",
    max_tokens=16000,
    additional_llm_args={"temperature": 0.3},
    verbose=True,
    interactive=False
)

# Create agents with detailed specifications
agent_specs = swarm.run(
    "Develop a comprehensive cybersecurity strategy for a mid-size company"
)

# Build agents from specifications
agents = swarm.create_agents_from_specs(agent_specs)

# Use the agents directly
for agent in agents:
    print(f"Agent: {agent.agent_name}")
    print(f"Description: {agent.description}")
    print(f"Model: {agent.model_name}")
    print(f"Max Loops: {agent.max_loops}")
    print("---")
```

## Best Practices

!!! tip "Task Definition"
    - Provide clear, specific task descriptions with context and constraints
    - Include expected output format and success criteria
    - Break complex tasks into smaller, manageable components
    - Consider task dependencies and coordination requirements
    - Use domain-specific terminology for better agent specialization

!!! note "Configuration"
    - Set appropriate `max_loops` based on task complexity (typically 1)
    - Use `verbose=True` during development for debugging
    - Choose the right `execution_type` for your use case:
        - Use `"return-agents"` for getting agent specifications as dictionary (default)
        - Use `"return-swarm-router-config"` for analyzing swarm architecture
        - Use `"return-agents-objects"` for getting agent objects created from specifications
    - Set `max_tokens` appropriately based on expected response length
    - Use `interactive=True` for real-time collaboration scenarios
    - Use `additional_llm_args` for passing custom parameters to the LLM

!!! note "Model Selection"
    - Choose appropriate `model_name` based on task requirements
    - Consider model capabilities, token limits, and cost
    - Use more powerful models (GPT-4.1, Claude-3) for complex reasoning
    - Use efficient models (GPT-4o-mini) for simple tasks
    - Balance performance with cost considerations
    - Test different models for optimal results

!!! note "Agent Design"
    - Leverage the comprehensive BOSS_SYSTEM_PROMPT for optimal agent creation
    - Use custom system prompts for domain-specific applications
    - Consider agent personality and role diversity for better collaboration
    - Set appropriate temperature values (0.1-0.7) for task requirements
    - Use `auto_generate_prompt=True` for dynamic prompt generation
    - Configure `max_tokens` based on expected response complexity

!!! note "Swarm Architecture"
    - Choose appropriate swarm types based on task requirements
    - Use `AgentRearrange` for dynamic task allocation
    - Use `MixtureOfAgents` for parallel processing
    - Use `GroupChat` for collaborative decision-making
    - Use `SequentialWorkflow` for linear task progression
    - Consider `HeavySwarm` for high-capacity processing

!!! warning "Error Handling"
    - Always wrap AutoSwarmBuilder calls in try-catch blocks
    - Implement appropriate fallback strategies for failures
    - Monitor error patterns and adjust configurations
    - Use comprehensive logging for debugging
    - Handle API rate limits and token limits gracefully

!!! info "Performance Optimization"
    - Use `batch_run()` for processing multiple similar tasks
    - Consider using `generate_router_config=True` for complex workflows
    - Monitor token usage with `max_tokens` parameter
    - Use appropriate `swarm_type` for your specific use case
    - Implement caching for repeated operations
    - Use parallel processing where appropriate

!!! info "Production Deployment"
    - Implement proper logging and monitoring
    - Use environment variables for sensitive configuration
    - Set up health checks and circuit breakers
    - Monitor resource usage and performance metrics
    - Implement graceful shutdown procedures
    - Use proper error reporting and alerting systems

### Best Practices for Error Handling

!!! warning "Always Handle Exceptions"
    - Wrap AutoSwarmBuilder calls in try-catch blocks
    - Log errors with appropriate detail levels
    - Implement appropriate fallback strategies
    - Monitor error patterns and adjust configurations

!!! tip "Debugging Configuration Issues"
    - Use `verbose=True` during development
    - Test with simple tasks first
    - Validate model names and API keys
    - Check token limits and rate limits

!!! note "Production Considerations"
    - Implement circuit breakers for external API calls
    - Use health checks to monitor system status
    - Set up proper logging and monitoring
    - Implement graceful shutdown procedures

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
