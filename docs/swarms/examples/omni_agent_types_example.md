# Omni Agent Types Tutorial

    End-to-end tutorial for `swarms.structs.omni_agent_types`.

    ## Prerequisites

    - Python 3.10+
    - `pip install -U swarms`

    ## Example

    ```python
    from typing import Callable
from swarms import Agent
from swarms.structs.omni_agent_types import AgentType, AgentListType

def fallback(task: str) -> str:
    return f"fallback: {task}"

agent: AgentType = Agent(agent_name="A", model_name="gpt-4.1")
agents: AgentListType = [agent, fallback]
    ```

    ## What this demonstrates

    - Correct import and initialization flow for `omni_agent_types`
    - Minimal execution path suitable for first integration tests
    - A baseline pattern to adapt for production use

    ## Related

    - Struct reference: [`swarms/structs/omni_agent_types.md`](../structs/omni_agent_types.md)
