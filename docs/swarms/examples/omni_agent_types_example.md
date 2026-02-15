# Omni Agent Types Tutorial

    End-to-end usage for `omni_agent_types`.

    ## Prerequisites

    - Python 3.10+
    - `pip install -U swarms`
    - Provider credentials configured when using hosted LLMs

    ## Example

    ```python
    from typing import Callable
from swarms.structs.omni_agent_types import AgentType, AgentListType

def run_any(agent: AgentType, task: str):
    if hasattr(agent, "run"):
        return agent.run(task)
    if isinstance(agent, Callable):
        return agent(task)
    return str(agent)

def batch_run(agents: AgentListType, task: str):
    return [run_any(agent, task) for agent in agents]
    ```

    ## What this demonstrates

    - Basic construction/import pattern for `omni_agent_types`
    - Minimal execution path you can adapt in production
    - Safe starting defaults for iteration

    ## Related

    - Struct reference: [`swarms/structs/omni_agent_types.md`](../structs/omni_agent_types.md)
