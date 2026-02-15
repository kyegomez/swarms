# Various Alt Swarms

    Reference documentation for `swarms.structs.various_alt_swarms`.

    ## Overview

    This module provides production utilities for `various alt swarms` in Swarms.

    ## Module Path

    ```python
    from swarms.structs.various_alt_swarms import ...
    ```

    ## Public API

    - Swarm classes: `CircularSwarm`, `StarSwarm`, `MeshSwarm`, `PyramidSwarm`, and communication patterns

    ## Quick Start

    ```python
    from swarms import Agent
from swarms.structs.various_alt_swarms import CircularSwarm

agents = [Agent(agent_name="A1", model_name="gpt-4.1"), Agent(agent_name="A2", model_name="gpt-4.1")]
swarm = CircularSwarm(agents=agents, output_type="list")
print(swarm.run(["Task 1", "Task 2"]))
    ```

    ## Tutorial

    See the runnable tutorial: [`swarms/examples/various_alt_swarms_example.md`](../examples/various_alt_swarms_example.md)

    ## Operational Notes

    - Validate credentials and model access before running LLM-backed examples.
    - Start with small inputs/tasks, then scale once behavior is verified.
