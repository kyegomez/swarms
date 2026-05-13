# AgentRegistry

`AgentRegistry` manages a collection of `Agent` instances keyed by each agent's
`agent_name`. Use it when an application needs to add, retrieve, update, delete,
or query agents at runtime while keeping registry operations thread-safe.

The registry stores the live `Agent` objects in `self.agents` and records
metadata for each registered agent in `AgentRegistrySchema` through
`AgentConfigSchema`.

## Quick Start

```python
from swarms import Agent
from swarms.structs.agent_registry import AgentRegistry

researcher = Agent(agent_name="Researcher")
writer = Agent(agent_name="Writer")

registry = AgentRegistry(
    name="Content Team",
    description="Agents used in the content workflow.",
    agents=[researcher],
)

registry.add(writer)

print(registry.list_agents())
# ["Researcher", "Writer"]

writer_agent = registry.get("Writer")
matching_agents = registry.query(
    lambda agent: agent.agent_name.endswith("er")
)

updated_writer = Agent(agent_name="Writer")
registry.update_agent("Writer", updated_writer)
registry.delete("Researcher")
```

## Constructor

```python
AgentRegistry(
    name="Agent Registry",
    description="A registry for managing agents.",
    agents=None,
    return_json=True,
    auto_save=False,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Human-readable registry name. |
| `description` | `str` | Human-readable registry description. |
| `agents` | `Optional[List[Agent]]` | Agents to register during initialization. |
| `return_json` | `bool` | Stores whether callers expect JSON output. |
| `auto_save` | `bool` | Stores whether automatic persistence should be enabled. |

## Methods

| Method | Description |
|--------|-------------|
| `add(agent)` | Adds an `Agent` using `agent.agent_name` as the registry key. Raises `ValueError` when that name already exists. |
| `add_many(agents)` | Adds a list of agents concurrently through `add`. |
| `delete(agent_name)` | Removes an agent by name. Raises `KeyError` when the name is missing. |
| `update_agent(agent_name, new_agent)` | Replaces the agent stored under `agent_name`. Raises `KeyError` when the name is missing. |
| `get(agent_name)` | Returns the agent stored under `agent_name`. Raises `KeyError` when the name is missing. |
| `list_agents()` | Returns all registered agent names. |
| `return_all_agents()` | Returns all registered `Agent` objects. |
| `query(condition=None)` | Returns all agents when `condition` is `None`; otherwise returns agents where `condition(agent)` is true. |
| `find_agent_by_name(agent_name)` | Searches the registry and returns the matching agent, or `None` when no match is found. |
| `find_agent_by_id(agent_id)` | Performs a direct dictionary lookup with the provided key. Since `add` stores agents by `agent_name`, prefer `get` or `find_agent_by_name` for normal lookups. |
| `agents_to_json()` | Serializes registered agents to a JSON string keyed by agent name. |
| `agent_to_py_model(agent)` | Converts an agent into `AgentConfigSchema` metadata and appends it to the registry schema. |

## Data Model

`AgentConfigSchema` stores metadata for each registered agent:

| Field | Type | Description |
|-------|------|-------------|
| `uuid` | `str` | Agent identifier from `agent.id`. |
| `name` | `str` | Agent name from `agent.agent_name`. |
| `description` | `str` | Agent description, or a default value when none is set. |
| `time_added` | `str` | UTC timestamp captured when the schema entry is created. |
| `config` | `Dict[Any, Any]` | Agent configuration from `agent.to_dict()`. |

`AgentRegistrySchema` stores registry-level metadata, including the registry
name, description, schema entries, creation timestamp, and number of agents.

## Usage Notes

- Agent names must be unique inside a registry because `agent.agent_name` is the
  storage key.
- `get`, `delete`, and `update_agent` raise `KeyError` for missing names.
- `add` raises `ValueError` for duplicate names.
- Use `query` for custom filtering without copying registry state manually.
- Use `agents_to_json` when a JSON snapshot of the registered agents is needed.
