# MakeASwarm Documentation

The `MakeASwarm` class is a comprehensive framework for **combining different swarm architectures** into complex, nested multi-agent systems. Unlike standard swarms that work with agents, MakeASwarm enables you to compose entire swarm architectures as components, creating hierarchical structures where swarms contain other swarms.

Full Path: `from swarms.structs.make_a_swarm import MakeASwarm`

## Overview

**Important**: MakeASwarm is designed for **combining swarm architectures**, not just creating simple agent workflows. While you can use regular swarms (SequentialWorkflow, ConcurrentWorkflow, etc.) for agent workflows, MakeASwarm enables you to combine different swarm types (BoardOfDirectors, HeavySwarm, HierarchicalSwarm, GroupChat, etc.) into complex, nested multi-architecture systems.

MakeASwarm is designed for **architectural composition** - combining different swarm types into unified workflows. Key capabilities:

- **Combine Swarm Architectures**: Use different swarm types (BoardOfDirectors, HeavySwarm, SequentialWorkflow, etc.) as components
- **Nested Swarm Structures**: Create hierarchies where swarms contain other swarms (e.g., BoardOfDirectors where each member is another BoardOfDirectors)
- **Mixed Architectures**: Combine different swarm types in a single workflow (e.g., HeavySwarm → BoardOfDirectors → GroupChat)
- **Execution Orchestration**: Control how different swarm architectures execute (sequential, concurrent, or dependency-based)
- **Export/Import**: Save complex swarm architecture combinations as reusable configurations
- **Factory Methods**: Dynamically create any swarm type from configuration

## Key Features

| Feature | Description |
|---------|-------------|
| **Component Registry** | Centralized registry for tracking agents and swarms by name |
| **Execution Modes** | Three execution modes: sequential, concurrent, and dependency-based |
| **Topological Sort** | Automatic dependency resolution using Kahn's algorithm |
| **Nested Swarms** | Support for swarms containing other swarms as agents |
| **JSON Import/Export** | Save and load swarm configurations as JSON files |
| **Automatic Module Generation** | Exported swarms automatically become importable Python modules |
| **Agent Factory** | Create agents from configuration dictionaries |
| **Swarm Factory** | Create swarms from configuration with support for all swarm types |
| **Cycle Detection** | Automatic detection of circular dependencies |

## Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"MakeASwarm"` | Name identifier for the swarm |
| `description` | `str` | `"A custom swarm created with MakeASwarm"` | Description of the swarm's purpose |
| `agents` | `List[Union[Agent, Callable]]` | `None` | List of agents (for BaseSwarm compatibility) |
| `max_loops` | `int` | `1` | Maximum number of execution loops |
| `execution_mode` | `Literal["sequential", "concurrent", "dependency"]` | `"sequential"` | Execution mode for the swarm |
| `execution_order` | `Union[List[str], Dict[str, List[str]]]` | `None` | Execution order specification |

## Quick Syntax Reference

### Basic Initialization

```python
from swarms.structs.make_a_swarm import MakeASwarm

swarm = MakeASwarm(
    name="MySwarm",                    # Optional: swarm identifier
    description="My custom swarm",     # Optional: description
    execution_mode="sequential",        # "sequential" | "concurrent" | "dependency"
    max_loops=1,                       # Optional: max execution loops
    execution_order=None                # Optional: can set here or later
)
```

### Adding Components (Swarm Architectures)

```python
# Add swarm architectures as components (primary use case)
from swarms.structs.board_of_directors_swarm import BoardOfDirectorsSwarm
from swarms.structs.heavy_swarm import HeavySwarm
from swarms.structs.groupchat import GroupChat

# Add different swarm architectures
swarm.add_component("research_swarm", heavy_swarm_instance)
swarm.add_component("decision_board", board_of_directors_swarm_instance)
swarm.add_component("discussion_group", group_chat_instance)

# You can also add agents or callables, but the power is in combining swarms
swarm.add_component("custom_func", lambda task: f"Processed: {task}")
```

### Creating Agents from Config

```python
# Base Agent
agent = swarm.create_agent({
    "agent_type": "agent",  # or "base" (required)
    "agent_name": "researcher",  # REQUIRED
    "system_prompt": "You are a research assistant",
    "model_name": "gpt-4o-mini",
    "max_loops": 1
})

# Specialized Agents (use lowercase agent_type)
# Supported types: "cot", "tot", "got", "aerasigma", "selfconsistency",
# "reflexion", "gkp", "judge", "crca", "ire", "reasoning-duo",
# "reasoning-router", "openai"

# ToT Agent Example
tot_agent = swarm.create_agent({
    "agent_type": "tot",
    "agent_name": "tot_agent",
    "model_name": "gpt-4o",
    "config": {
        "max_depth": 3,
        "branch_factor": 2,
        "beam_width": 3,
        "search_strategy": "beam"
    }
})

# Reflexion Agent Example
reflexion_agent = swarm.create_agent({
    "agent_type": "reflexion",
    "agent_name": "reflexion_agent",
    "model_name": "openai/o1",
    "max_loops": 3,
    "memory_capacity": 100
})

# SelfConsistency Agent Example
consistency_agent = swarm.create_agent({
    "agent_type": "selfconsistency",
    "agent_name": "consistency_agent",
    "model_name": "gpt-4o",
    "num_samples": 3
})

# ReasoningDuo Example
reasoning_duo = swarm.create_agent({
    "agent_type": "reasoning-duo",
    "agent_name": "reasoning_duo",
    "model_name": "gpt-4o-mini",
    "reasoning_model_name": "claude-3-5-sonnet-20240620"
})
```

### Creating Swarms from Config

```python
# Create nested swarm
nested_swarm = swarm.create_swarm(
    "analysis",                        # Component name
    "SequentialWorkflow",              # Swarm type
    {
        "name": "AnalysisSwarm",
        "agents": [agent1, agent2],
        "max_loops": 1
    }
)

# Supported swarm types:
# "SequentialWorkflow", "ConcurrentWorkflow", "GroupChat", "HeavySwarm",
# "HierarchicalSwarm", "MixtureOfAgents", "MajorityVoting", "MALT",
# "CouncilAsAJudge", "InteractiveGroupChat", "MultiAgentRouter",
# "BoardOfDirectorsSwarm", "BatchedGridWorkflow", "AgentRearrange",
# "RoundRobinSwarm", "SpreadSheetSwarm", "SwarmRouter", "GraphWorkflow",
# "AutoSwarmBuilder", "HybridHierarchicalClusterSwarm", "AOP",
# "SelfMoASeq", "SocialAlgorithms"

# BoardOfDirectorsSwarm Example
bod_swarm = swarm.create_swarm(
    "board",
    "BoardOfDirectorsSwarm",
    {
        "name": "MyBoard",
        "agents": [agent1, agent2],
        "max_loops": 1,
        "board_model_name": "gpt-4o-mini"
    }
)
```

### Setting Execution Order

```python
# Sequential Mode (simple list)
swarm.set_execution_order(["agent1", "agent2", "agent3"])

# Concurrent Mode (same list, but execution_mode="concurrent")
swarm.execution_mode = "concurrent"
swarm.set_execution_order(["agent1", "agent2", "agent3"])

# Dependency Mode (dictionary with dependencies)
swarm.execution_mode = "dependency"
swarm.set_execution_order({
    "agent1": [],                    # No dependencies - runs first
    "agent2": ["agent1"],            # Depends on agent1
    "agent3": ["agent1", "agent2"]   # Depends on both
})
```

### Complete Workflow Pattern (Combining Swarm Architectures)

```python
# 1. Initialize
swarm = MakeASwarm(name="MySwarm", execution_mode="sequential")

# 2. Create different swarm architectures
from swarms.structs.heavy_swarm import HeavySwarm
from swarms.structs.board_of_directors_swarm import BoardOfDirectorsSwarm
from swarms import Agent

# Create a HeavySwarm
research_swarm = HeavySwarm(name="Research", show_dashboard=False, max_loops=1)

# Create a BoardOfDirectorsSwarm
board_agents = [
    Agent(agent_name="director1", system_prompt="Director 1", model_name="gpt-4o-mini"),
    Agent(agent_name="director2", system_prompt="Director 2", model_name="gpt-4o-mini")
]
decision_board = BoardOfDirectorsSwarm(
    name="DecisionBoard",
    agents=board_agents,
    max_loops=1,
    board_model_name="gpt-4o-mini"
)

# 3. Add swarm architectures as components
swarm.add_component("research", research_swarm)
swarm.add_component("decision", decision_board)

# 4. Set execution order
swarm.set_execution_order(["research", "decision"])

# 5. Build
swarm.build()

# 6. Run
result = swarm.run("Research and make strategic decisions")

# 7. Export (optional)
swarm.export_to_json("my_swarm.json")
```

### Export and Import

```python
# Export to JSON (automatically generates Python module)
swarm.export_to_json("my_swarm.json")
# Creates: my_swarm.json and swarms/exports/my_swarm.py

# Import using auto-generated module (recommended)
from swarms.exports import my_swarm
result = my_swarm.swarm.run("task")

# Or import using load_from_json
swarm = MakeASwarm()
swarm.load_from_json("my_swarm.json")
result = swarm.run("task")
```

## Execution Modes

### Sequential Mode

Components execute one after another in the specified order. Each component receives the task and optionally the result from the previous component.

```python
execution_order = ["agent1", "agent2", "agent3"]
```

### Concurrent Mode

All components execute simultaneously in parallel. Useful for independent tasks that don't depend on each other.

```python
execution_order = ["agent1", "agent2", "agent3"]
execution_mode = "concurrent"
```

### Dependency Mode

Components execute based on a dependency graph. Uses topological sorting to determine execution order, with independent components executing in parallel.

```python
execution_order = {
    "agent1": [],
    "agent2": ["agent1"],
    "agent3": ["agent1", "agent2"]
}
execution_mode = "dependency"
```

## Core Methods

### `add_component(name: str, component: Union[Agent, BaseSwarm]) -> None`

Add a component (agent or swarm) to the registry.

**Parameters:**
- `name` (str): Name identifier for the component
- `component` (Union[Agent, BaseSwarm]): The component to add

**Example:**
```python
swarm.add_component("researcher", researcher_agent)
swarm.add_component("analyzer", analyzer_swarm)
```

### `create_agent(config: Dict[str, Any]) -> Agent`

Create an agent from a configuration dictionary.

**Parameters:**
- `config` (Dict[str, Any]): Agent configuration dictionary

**Required fields:**
- `agent_name` (str): Name of the agent
- `system_prompt` (str): System prompt for the agent

**Optional fields:**
- All other Agent constructor parameters (model_name, max_loops, tools, etc.)

**Example:**
```python
agent_config = {
    "agent_name": "researcher",
    "system_prompt": "You are a research assistant.",
    "model_name": "gpt-4o-mini",
    "max_loops": 1
}
agent = swarm.create_agent(agent_config)
swarm.add_component("researcher", agent)
```

### `create_swarm(name: str, swarm_type: Union[str, type], config: Dict[str, Any]) -> BaseSwarm`

Create a swarm from a configuration dictionary.

**Parameters:**
- `name` (str): Name identifier for the swarm
- `swarm_type` (Union[str, type]): Swarm type (string or class)
- `config` (Dict[str, Any]): Swarm configuration dictionary

**Supported Swarm Types:**
- `"SequentialWorkflow"`
- `"ConcurrentWorkflow"`
- `"GroupChat"`
- `"HeavySwarm"`
- `"HierarchicalSwarm"`
- `"MixtureOfAgents"`
- `"MajorityVoting"`
- `"MALT"`
- `"CouncilAsAJudge"`
- `"InteractiveGroupChat"`
- `"MultiAgentRouter"`
- `"BoardOfDirectorsSwarm"`
- `"BatchedGridWorkflow"`
- Custom swarm classes (passed as type)

**Example:**
```python
swarm_config = {
    "name": "analysis_swarm",
    "description": "Analysis swarm",
    "agents": [agent1, agent2],
    "max_loops": 1
}
analysis_swarm = swarm.create_swarm("analysis", "SequentialWorkflow", swarm_config)
swarm.add_component("analysis", analysis_swarm)
```

### `set_execution_order(order: Union[List[str], Dict[str, List[str]]]) -> None`

Set the execution order for components.

**Parameters:**
- `order` (Union[List[str], Dict[str, List[str]]]): Execution order
  - `List[str]`: Simple sequential order
  - `Dict[str, List[str]]`: Dependency graph where keys are component names and values are lists of dependencies

**Example:**
```python
# Sequential order
swarm.set_execution_order(["agent1", "agent2", "agent3"])

# Dependency graph
swarm.set_execution_order({
    "agent1": [],
    "agent2": ["agent1"],
    "agent3": ["agent2"]
})
```

### `build() -> None`

Build the final swarm structure. Validates the configuration and prepares the swarm for execution.

**Raises:**
- `ComponentNotFoundError`: If a component in execution_order is not found in registry

### `run(task: Optional[str] = None, *args, **kwargs) -> Any`

Run the swarm with the specified task.

**Parameters:**
- `task` (Optional[str]): Task to execute
- `*args`: Additional positional arguments
- `**kwargs`: Additional keyword arguments

**Returns:**
- Execution results (format depends on execution_mode and output_type)

**Example:**
```python
result = swarm.run("Analyze this data and provide insights")
```

### `export_to_json(filepath: str) -> None`

Export the swarm configuration to a JSON file and automatically generate an importable Python module.

**Parameters:**
- `filepath` (str): Path to the JSON file to create

**Raises:**
- `IOError`: If file cannot be written

**Note:**
When you export to a JSON file (e.g., `my_swarm.json`), MakeASwarm automatically creates a Python module at `swarms/exports/my_swarm.py` that can be imported directly. The module name is derived from the JSON filename.

**Example:**
```python
swarm.export_to_json("my_swarm_config.json")
# This creates:
# - my_swarm_config.json (configuration file)
# - swarms/exports/my_swarm_config.py (importable module)

# You can now import and use it:
from swarms.exports import my_swarm_config
result = my_swarm_config.swarm.run("your task here")
```

### `load_from_json(filepath: str) -> None`

Load swarm configuration from a JSON file.

**Parameters:**
- `filepath` (str): Path to the JSON file to load

**Raises:**
- `FileNotFoundError`: If file does not exist
- `ValueError`: If configuration is invalid

**Example:**
```python
swarm.load_from_json("my_swarm_config.json")
```

## Getting Started

### Example 1: Combining Different Swarm Architectures

Create a workflow that combines different swarm types - a HeavySwarm for research, followed by a BoardOfDirectors for decision-making:

```python
from swarms.structs.make_a_swarm import MakeASwarm
from swarms.structs.heavy_swarm import HeavySwarm
from swarms.structs.board_of_directors_swarm import BoardOfDirectorsSwarm
from swarms import Agent

# Create a HeavySwarm for comprehensive research
research_swarm = HeavySwarm(
    name="ResearchSwarm",
    show_dashboard=False,
    max_loops=1
)

# Create a BoardOfDirectorsSwarm for decision-making
decision_agents = [
    Agent(agent_name="strategist", system_prompt="Strategic planning expert", model_name="gpt-4o-mini"),
    Agent(agent_name="analyst", system_prompt="Data analysis expert", model_name="gpt-4o-mini"),
    Agent(agent_name="risk_manager", system_prompt="Risk assessment expert", model_name="gpt-4o-mini")
]

decision_board = BoardOfDirectorsSwarm(
    name="DecisionBoard",
    agents=decision_agents,
    max_loops=1,
    board_model_name="gpt-4o-mini"
)

# Combine both swarm architectures
main_swarm = MakeASwarm(
    name="ResearchAndDecisionSwarm",
    description="Combines HeavySwarm research with BoardOfDirectors decision-making",
    execution_mode="sequential"
)

# Add swarm architectures as components
main_swarm.add_component("research", research_swarm)
main_swarm.add_component("decision", decision_board)

# Set execution order: research first, then decision
main_swarm.set_execution_order(["research", "decision"])

main_swarm.build()
result = main_swarm.run("Research market trends and make strategic recommendations")
```

### Example 2: Concurrent Swarm Architectures

Execute multiple different swarm architectures in parallel:

```python
from swarms.structs.make_a_swarm import MakeASwarm
from swarms.structs.groupchat import GroupChat
from swarms.structs.mixture_of_agents import MixtureOfAgents
from swarms.structs.majority_voting import MajorityVoting
from swarms import Agent

# Create different swarm architectures
group_chat_agents = [
    Agent(agent_name="expert1", system_prompt="Expert in domain A", model_name="gpt-4o-mini"),
    Agent(agent_name="expert2", system_prompt="Expert in domain B", model_name="gpt-4o-mini"),
    Agent(agent_name="expert3", system_prompt="Expert in domain C", model_name="gpt-4o-mini")
]

group_chat = GroupChat(
    name="ExpertDiscussion",
    agents=group_chat_agents,
    max_loops=3
)

mixture_agents = [
    Agent(agent_name="analyst1", system_prompt="Data analyst", model_name="gpt-4o-mini"),
    Agent(agent_name="analyst2", system_prompt="Data analyst", model_name="gpt-4o-mini")
]

mixture_swarm = MixtureOfAgents(
    name="AnalysisMixture",
    agents=mixture_agents,
    max_loops=1
)

voting_agents = [
    Agent(agent_name="reviewer1", system_prompt="Quality reviewer", model_name="gpt-4o-mini"),
    Agent(agent_name="reviewer2", system_prompt="Quality reviewer", model_name="gpt-4o-mini"),
    Agent(agent_name="reviewer3", system_prompt="Quality reviewer", model_name="gpt-4o-mini")
]

voting_swarm = MajorityVoting(
    name="QualityVoting",
    agents=voting_agents,
    max_loops=1
)

# Combine all three swarm architectures to run concurrently
main_swarm = MakeASwarm(
    name="ParallelArchitectures",
    execution_mode="concurrent"
)

main_swarm.add_component("discussion", group_chat)
main_swarm.add_component("analysis", mixture_swarm)
main_swarm.add_component("voting", voting_swarm)

main_swarm.set_execution_order(["discussion", "analysis", "voting"])
main_swarm.build()

# All swarm architectures execute simultaneously
results = main_swarm.run("Analyze the market trends from multiple perspectives")
```

### Example 3: Dependency-Based Swarm Architecture Pipeline

Create a workflow where different swarm architectures depend on each other:

```python
from swarms.structs.make_a_swarm import MakeASwarm
from swarms.structs.sequential_workflow import SequentialWorkflow
from swarms.structs.board_of_directors_swarm import BoardOfDirectorsSwarm
from swarms.structs.concurrent_workflow import ConcurrentWorkflow
from swarms import Agent

# Create a SequentialWorkflow for data collection
collection_agents = [
    Agent(agent_name="collector1", system_prompt="Collect data from source A", model_name="gpt-4o-mini"),
    Agent(agent_name="collector2", system_prompt="Collect data from source B", model_name="gpt-4o-mini")
]

collection_swarm = SequentialWorkflow(
    name="DataCollection",
    agents=collection_agents,
    max_loops=1
)

# Create a BoardOfDirectorsSwarm for analysis (depends on collection)
analysis_agents = [
    Agent(agent_name="analyst1", system_prompt="Data analyst", model_name="gpt-4o-mini"),
    Agent(agent_name="analyst2", system_prompt="Statistical analyst", model_name="gpt-4o-mini")
]

analysis_board = BoardOfDirectorsSwarm(
    name="AnalysisBoard",
    agents=analysis_agents,
    max_loops=1,
    board_model_name="gpt-4o-mini"
)

# Create a ConcurrentWorkflow for reporting (depends on analysis)
report_agents = [
    Agent(agent_name="writer", system_prompt="Report writer", model_name="gpt-4o-mini"),
    Agent(agent_name="visualizer", system_prompt="Data visualizer", model_name="gpt-4o-mini")
]

report_swarm = ConcurrentWorkflow(
    name="ReportGeneration",
    agents=report_agents,
    max_loops=1
)

# Combine swarm architectures with dependencies
main_swarm = MakeASwarm(
    name="DataPipeline",
    execution_mode="dependency"
)

main_swarm.add_component("collection", collection_swarm)
main_swarm.add_component("analysis", analysis_board)
main_swarm.add_component("reporting", report_swarm)

# Define dependencies: analysis depends on collection, reporting depends on analysis
main_swarm.set_execution_order({
    "collection": [],  # No dependencies - runs first
    "analysis": ["collection"],  # Depends on collection
    "reporting": ["analysis"]  # Depends on analysis
})

main_swarm.build()
results = main_swarm.run("Process quarterly sales data")
```

### Example 4: Nested Swarm Architectures

Create a hierarchical structure where swarm architectures contain other swarm architectures:

```python
from swarms.structs.make_a_swarm import MakeASwarm
from swarms.structs.hierarchical_swarm import HierarchicalSwarm
from swarms.structs.groupchat import GroupChat
from swarms.structs.board_of_directors_swarm import BoardOfDirectorsSwarm
from swarms import Agent

# Create a GroupChat swarm for collaborative discussion
group_agents = [
    Agent(agent_name="expert1", system_prompt="Domain expert 1", model_name="gpt-4o-mini"),
    Agent(agent_name="expert2", system_prompt="Domain expert 2", model_name="gpt-4o-mini")
]

discussion_swarm = GroupChat(
    name="ExpertDiscussion",
    agents=group_agents,
    max_loops=3
)

# Create a BoardOfDirectorsSwarm for decision-making
board_agents = [
    Agent(agent_name="director1", system_prompt="Board director 1", model_name="gpt-4o-mini"),
    Agent(agent_name="director2", system_prompt="Board director 2", model_name="gpt-4o-mini")
]

decision_swarm = BoardOfDirectorsSwarm(
    name="DecisionBoard",
    agents=board_agents,
    max_loops=1,
    board_model_name="gpt-4o-mini"
)

# Create a HierarchicalSwarm that contains both swarms
hierarchical_agents = [discussion_swarm, decision_swarm]

hierarchical_swarm = HierarchicalSwarm(
    name="HierarchicalStructure",
    agents=hierarchical_agents,
    max_loops=1
)

# Use MakeASwarm to orchestrate the hierarchical swarm
main_swarm = MakeASwarm(
    name="NestedArchitectures",
    execution_mode="sequential"
)

main_swarm.add_component("hierarchical", hierarchical_swarm)
main_swarm.set_execution_order(["hierarchical"])
main_swarm.build()

result = main_swarm.run("Collaborate and make decisions on the project")
```

### Example 5: Nested Board of Directors Architecture

Create a hierarchical structure where BoardOfDirectors swarms contain other BoardOfDirectors swarms:

```python
from swarms.structs.make_a_swarm import MakeASwarm
from swarms.structs.board_of_directors_swarm import BoardOfDirectorsSwarm
from swarms import Agent

# Create Level 2: Department Boards (BoardOfDirectors swarms)
dept1_agents = [
    Agent(agent_name="dept1_agent1", system_prompt="Department 1 specialist", model_name="gpt-4o-mini"),
    Agent(agent_name="dept1_agent2", system_prompt="Department 1 specialist", model_name="gpt-4o-mini")
]

dept1_board = BoardOfDirectorsSwarm(
    name="Department1Board",
    agents=dept1_agents,
    max_loops=1,
    board_model_name="gpt-4o-mini"
)

dept2_agents = [
    Agent(agent_name="dept2_agent1", system_prompt="Department 2 specialist", model_name="gpt-4o-mini"),
    Agent(agent_name="dept2_agent2", system_prompt="Department 2 specialist", model_name="gpt-4o-mini")
]

dept2_board = BoardOfDirectorsSwarm(
    name="Department2Board",
    agents=dept2_agents,
    max_loops=1,
    board_model_name="gpt-4o-mini"
)

# Create Level 1: Top-level Board where members are other BoardOfDirectors swarms
top_board = BoardOfDirectorsSwarm(
    name="TopLevelBoard",
    agents=[dept1_board, dept2_board],  # Board members are themselves boards!
    max_loops=1,
    board_model_name="gpt-4o-mini"
)

# Use MakeASwarm to orchestrate the nested board structure
main_swarm = MakeASwarm(
    name="NestedBoardArchitecture",
    execution_mode="sequential"
)

main_swarm.add_component("top_board", top_board)
main_swarm.set_execution_order(["top_board"])
main_swarm.build()

result = main_swarm.run("Make strategic decisions for the company")
# This creates a 2-level hierarchy: Top Board → Department Boards → Agents
```

### Example 6: Creating Swarm Architectures from Config

Create different swarm architectures dynamically from configuration and combine them:

```python
from swarms.structs.make_a_swarm import MakeASwarm
from swarms import Agent

# Create base agents first
agent1 = Agent(agent_name="agent1", system_prompt="Agent 1", model_name="gpt-4o-mini")
agent2 = Agent(agent_name="agent2", system_prompt="Agent 2", model_name="gpt-4o-mini")
agent3 = Agent(agent_name="agent3", system_prompt="Agent 3", model_name="gpt-4o-mini")

main_swarm = MakeASwarm(name="ConfigBasedArchitectures", execution_mode="sequential")

# Create a SequentialWorkflow swarm from config
sequential_config = {
    "name": "SequentialSwarm",
    "agents": [agent1, agent2],
    "max_loops": 1
}

sequential_swarm = main_swarm.create_swarm("sequential", "SequentialWorkflow", sequential_config)
main_swarm.add_component("sequential", sequential_swarm)

# Create a BoardOfDirectorsSwarm from config
board_config = {
    "name": "DecisionBoard",
    "agents": [agent2, agent3],
    "max_loops": 1,
    "board_model_name": "gpt-4o-mini"
}

board_swarm = main_swarm.create_swarm("board", "BoardOfDirectorsSwarm", board_config)
main_swarm.add_component("board", board_swarm)

# Combine both swarm architectures
main_swarm.set_execution_order(["sequential", "board"])
main_swarm.build()

result = main_swarm.run("Process data sequentially then make board decisions")
```

### Example 7: Complex Multi-Architecture Composition

Combine multiple different swarm architectures in a complex workflow:

```python
from swarms.structs.make_a_swarm import MakeASwarm
from swarms.structs.heavy_swarm import HeavySwarm
from swarms.structs.board_of_directors_swarm import BoardOfDirectorsSwarm
from swarms.structs.groupchat import GroupChat
from swarms.structs.sequential_workflow import SequentialWorkflow
from swarms import Agent

# Create a HeavySwarm for comprehensive analysis
heavy_swarm = HeavySwarm(
    name="AnalysisHeavySwarm",
    show_dashboard=False,
    max_loops=1
)

# Create a BoardOfDirectorsSwarm for strategic decisions
board_agents = [
    Agent(agent_name="strategist", system_prompt="Strategic planner", model_name="gpt-4o-mini"),
    Agent(agent_name="analyst", system_prompt="Business analyst", model_name="gpt-4o-mini")
]

decision_board = BoardOfDirectorsSwarm(
    name="StrategicBoard",
    agents=board_agents,
    max_loops=1,
    board_model_name="gpt-4o-mini"
)

# Create a GroupChat for collaborative discussion
group_agents = [
    Agent(agent_name="expert1", system_prompt="Domain expert", model_name="gpt-4o-mini"),
    Agent(agent_name="expert2", system_prompt="Domain expert", model_name="gpt-4o-mini")
]

discussion_group = GroupChat(
    name="ExpertGroup",
    agents=group_agents,
    max_loops=3
)

# Create a SequentialWorkflow for final processing
processing_agents = [
    Agent(agent_name="processor1", system_prompt="Data processor", model_name="gpt-4o-mini"),
    Agent(agent_name="processor2", system_prompt="Data processor", model_name="gpt-4o-mini")
]

processing_workflow = SequentialWorkflow(
    name="ProcessingWorkflow",
    agents=processing_agents,
    max_loops=1
)

# Combine all four different swarm architectures
main_swarm = MakeASwarm(
    name="ComplexMultiArchitecture",
    execution_mode="dependency"
)

main_swarm.add_component("analysis", heavy_swarm)
main_swarm.add_component("decision", decision_board)
main_swarm.add_component("discussion", discussion_group)
main_swarm.add_component("processing", processing_workflow)

# Define complex dependencies
main_swarm.set_execution_order({
    "analysis": [],  # Runs first
    "decision": ["analysis"],  # Depends on analysis
    "discussion": ["analysis"],  # Parallel with decision, depends on analysis
    "processing": ["decision", "discussion"]  # Depends on both decision and discussion
})

main_swarm.build()
results = main_swarm.run("Comprehensive business analysis and decision-making")
```

## JSON Import/Export

### Exporting a Swarm Configuration

Save your swarm configuration to a JSON file for reuse. When you export, MakeASwarm automatically generates an importable Python module:

```python
from swarms import Agent, MakeASwarm

# Create and configure swarm
swarm = MakeASwarm(name="MySwarm", execution_mode="sequential")

researcher = Agent(
    agent_name="researcher",
    system_prompt="Research assistant",
    model_name="gpt-4o-mini"
)

writer = Agent(
    agent_name="writer",
    system_prompt="Writer",
    model_name="gpt-4o-mini"
)

swarm.add_component("researcher", researcher)
swarm.add_component("writer", writer)
swarm.set_execution_order(["researcher", "writer"])
swarm.build()

# Export to JSON (also generates importable module)
swarm.export_to_json("my_swarm_config.json")
# Creates: my_swarm_config.json and swarms/exports/my_swarm_config.py
```

### Using Auto-Generated Import Modules

When you export a swarm, it automatically becomes an importable Python module:

```python
# After exporting to "my_swarm_config.json", you can import it:
from swarms.exports import my_swarm_config

# Use the swarm directly
result = my_swarm_config.swarm.run("Research and write about AI")

# Access metadata
print(my_swarm_config.name)  # "MySwarm"
print(my_swarm_config.description)  # Swarm description
print(my_swarm_config.config_path)  # Path to JSON config file
```

**Module Structure:**
- `swarm`: The loaded MakeASwarm instance, ready to use
- `name`: The swarm's name
- `description`: The swarm's description
- `config_path`: Path to the JSON configuration file

**Benefits:**
- **Reusability**: Import your swarms in any Python project
- **Version Control**: Track swarm configurations alongside code
- **Easy Sharing**: Share swarms by sharing the JSON file and generated module
- **No Manual Setup**: The module is automatically generated and ready to use

### Loading a Swarm Configuration

You can load a previously saved swarm configuration in two ways:

**Method 1: Using `load_from_json()`**
```python
from swarms import MakeASwarm

# Create new swarm instance
swarm = MakeASwarm()

# Load configuration from JSON
swarm.load_from_json("my_swarm_config.json")

# Swarm is ready to use
result = swarm.run("Execute the loaded workflow")
```

**Method 2: Using the Auto-Generated Import Module (Recommended)**
```python
# Simply import the auto-generated module
from swarms.exports import my_swarm_config

# The swarm is already loaded and ready to use
result = my_swarm_config.swarm.run("Execute the loaded workflow")

# Access metadata if needed
print(f"Using swarm: {my_swarm_config.name}")
print(f"Description: {my_swarm_config.description}")
```

The auto-generated module approach is recommended because:
- No need to manually create a MakeASwarm instance
- The swarm is automatically loaded and ready to use
- Cleaner, more Pythonic import syntax
- Easy to share and reuse across projects

### JSON Configuration Format

The JSON configuration file supports both agents and swarm architectures. Here's an example combining swarm architectures:

```json
{
  "name": "MyMultiArchitectureSwarm",
  "description": "Combines HeavySwarm and BoardOfDirectors",
  "execution_mode": "sequential",
  "execution_order": ["research", "decision"],
  "max_loops": 1,
  "components": {
    "research": {
      "type": "HeavySwarm",
      "config": {
        "name": "ResearchSwarm",
        "show_dashboard": false,
        "max_loops": 1
      }
    },
    "decision": {
      "type": "BoardOfDirectorsSwarm",
      "config": {
        "name": "DecisionBoard",
        "max_loops": 1,
        "board_model_name": "gpt-4o-mini"
      }
    }
  }
}
```

## Advanced Usage

### Custom Swarm Types

You can use custom swarm classes by passing the class type:

```python
from swarms import MakeASwarm
from my_custom_swarm import MyCustomSwarm

swarm = MakeASwarm()

custom_config = {
    "name": "CustomSwarm",
    "agents": [...],
    "max_loops": 1
}

custom_swarm = swarm.create_swarm(
    "custom",
    MyCustomSwarm,  # Pass class type
    custom_config
)

swarm.add_component("custom", custom_swarm)
```

### Complex Dependency Graphs with Swarm Architectures

Create complex workflows where different swarm architectures depend on each other:

```python
from swarms.structs.make_a_swarm import MakeASwarm
from swarms.structs.heavy_swarm import HeavySwarm
from swarms.structs.board_of_directors_swarm import BoardOfDirectorsSwarm
from swarms.structs.groupchat import GroupChat
from swarms.structs.sequential_workflow import SequentialWorkflow
from swarms import Agent

swarm = MakeASwarm(execution_mode="dependency")

# Create different swarm architectures
heavy_swarm = HeavySwarm(name="Research", show_dashboard=False, max_loops=1)
swarm.add_component("research", heavy_swarm)

board_agents = [Agent(agent_name=f"director{i}", system_prompt=f"Director {i}", model_name="gpt-4o-mini") for i in range(2)]
decision_board = BoardOfDirectorsSwarm(name="Decision", agents=board_agents, max_loops=1, board_model_name="gpt-4o-mini")
swarm.add_component("decision", decision_board)

group_agents = [Agent(agent_name=f"expert{i}", system_prompt=f"Expert {i}", model_name="gpt-4o-mini") for i in range(2)]
discussion_group = GroupChat(name="Discussion", agents=group_agents, max_loops=3)
swarm.add_component("discussion", discussion_group)

seq_agents = [Agent(agent_name=f"processor{i}", system_prompt=f"Processor {i}", model_name="gpt-4o-mini") for i in range(2)]
processing_workflow = SequentialWorkflow(name="Processing", agents=seq_agents, max_loops=1)
swarm.add_component("processing", processing_workflow)

# Complex dependency graph with swarm architectures
swarm.set_execution_order({
    "research": [],  # Level 0 - runs first
    "decision": ["research"],  # Level 1 - depends on research
    "discussion": ["research"],  # Level 1 - parallel with decision, depends on research
    "processing": ["decision", "discussion"]  # Level 2 - depends on both decision and discussion
})

swarm.build()
results = swarm.run("Execute complex multi-architecture workflow")
```

### Error Handling

Handle errors gracefully:

```python
from swarms import MakeASwarm, ComponentNotFoundError, CycleDetectionError

swarm = MakeASwarm()

try:
    swarm.set_execution_order({
        "agent1": ["agent2"],
        "agent2": ["agent1"]  # Circular dependency
    })
    swarm.build()
except CycleDetectionError as e:
    print(f"Cycle detected: {e}")

try:
    swarm.set_execution_order(["nonexistent_agent"])
    swarm.build()
except ComponentNotFoundError as e:
    print(f"Component not found: {e}")
```

## Best Practices

1. **Focus on Swarm Architecture Composition**: MakeASwarm is designed for combining different swarm types (HeavySwarm, BoardOfDirectors, GroupChat, etc.), not just simple agent workflows. Use it to create complex multi-architecture systems.

2. **Name Swarm Components Clearly**: Use descriptive names for swarm architectures to make execution orders readable (e.g., "research_heavy_swarm", "decision_board", "discussion_group").

3. **Leverage Nested Structures**: Take advantage of MakeASwarm's ability to nest swarm architectures - create BoardOfDirectors where members are other swarms, or HierarchicalSwarm containing multiple swarm types.

4. **Validate Before Building**: Always call `build()` before `run()` to catch configuration errors early.

5. **Use Dependency Mode for Complex Architectures**: When swarm architectures have dependencies, use dependency mode for optimal parallelization.

6. **Export Complex Configurations**: Save your multi-architecture swarm configurations to JSON for reuse and version control. The exported swarms automatically become importable Python modules.

7. **Combine Different Swarm Types**: Mix and match different swarm architectures (HeavySwarm for research, BoardOfDirectors for decisions, GroupChat for collaboration) to leverage each architecture's strengths.

8. **Handle Errors**: Wrap swarm operations in try-except blocks to handle errors gracefully.

9. **Test Incrementally**: Start with combining 2-3 swarm architectures and gradually add complexity.

10. **Document Architecture Combinations**: Use comments to explain why you're combining specific swarm types and how they work together.

## Supported Swarm Types

MakeASwarm supports all standard swarm types:

- SequentialWorkflow
- ConcurrentWorkflow
- GroupChat
- HeavySwarm
- HierarchicalSwarm
- MixtureOfAgents
- MajorityVoting
- MALT
- CouncilAsAJudge
- InteractiveGroupChat
- MultiAgentRouter
- BoardOfDirectorsSwarm
- BatchedGridWorkflow
- Custom swarm classes

## Mathematical Concepts

MakeASwarm uses several computer science concepts:

- **Graph Theory**: Topological sorting (Kahn's algorithm) for dependency resolution
- **Composite Pattern**: Nested swarms containing other swarms
- **Factory Pattern**: Dynamic creation of agents and swarms
- **Registry Pattern**: Name-based component lookup and reuse

## Troubleshooting

### Common Issues

**Issue**: `ComponentNotFoundError` when building
- **Solution**: Ensure all components in execution_order are registered using `add_component()`

**Issue**: `CycleDetectionError` in dependency mode
- **Solution**: Check your dependency graph for circular dependencies

**Issue**: Agents not executing in expected order
- **Solution**: Verify execution_mode matches your execution_order format (list vs dict)

**Issue**: JSON import fails
- **Solution**: Validate JSON structure matches expected format and all required fields are present

## See Also

- [Agent Documentation](./agent.md) - For creating individual agents
- [SwarmRouter Documentation](./swarm_router.md) - For routing tasks to different swarm types
- [SequentialWorkflow Documentation](./sequential_workflow.md) - For sequential execution patterns
- [ConcurrentWorkflow Documentation](./concurrentworkflow.md) - For parallel execution patterns
- [BoardOfDirectors Documentation](./BoardOfDirectors.md) - For board-based decision making

