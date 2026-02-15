# Collaborative Utils

    Reference documentation for `swarms.structs.collaborative_utils`.

    ## Overview

    This module provides production utilities for `collaborative utils` in Swarms.

    ## Module Path

    ```python
    from swarms.structs.collaborative_utils import ...
    ```

    ## Public API

    - `talk_to_agent(current_agent, agents, task, agent_name, max_loops, output_type)`

    ## Quick Start

    ```python
    from swarms import Agent
from swarms.structs.collaborative_utils import talk_to_agent

alice = Agent(agent_name="alice", model_name="gpt-4.1")
bob = Agent(agent_name="bob", model_name="gpt-4.1")
print(talk_to_agent(current_agent=alice, agents=[alice, bob], task="Debate caching strategy", agent_name="bob", max_loops=2))
    ```

    ## Tutorial

    See the runnable tutorial: [`swarms/examples/collaborative_utils_example.md`](../examples/collaborative_utils_example.md)

    ## Operational Notes

    - Validate credentials and model access before running LLM-backed examples.
    - Start with small inputs/tasks, then scale once behavior is verified.
