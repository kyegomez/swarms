# Omni Agent Types

    Reference documentation for `swarms.structs.omni_agent_types`.

    ## Overview

    This module provides production utilities for `omni agent types` in Swarms.

    ## Module Path

    ```python
    from swarms.structs.omni_agent_types import ...
    ```

    ## Public API

    - `AgentType` and `AgentListType` aliases for shared type-safe APIs

    ## Quick Start

    ```python
    from typing import Callable
from swarms import Agent
from swarms.structs.omni_agent_types import AgentType, AgentListType

def fallback(task: str) -> str:
    return f"fallback: {task}"

agent: AgentType = Agent(agent_name="A", model_name="gpt-4.1")
agents: AgentListType = [agent, fallback]
    ```

    ## Tutorial

    See the runnable tutorial: [`swarms/examples/omni_agent_types_example.md`](../examples/omni_agent_types_example.md)

    ## Operational Notes

    - Validate credentials and model access before running LLM-backed examples.
    - Start with small inputs/tasks, then scale once behavior is verified.
