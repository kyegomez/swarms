# Various Alt Swarms Tutorial

    End-to-end tutorial for `swarms.structs.various_alt_swarms`.

    ## Prerequisites

    - Python 3.10+
    - `pip install -U swarms`

    ## Example

    ```python
    from swarms import Agent
from swarms.structs.various_alt_swarms import CircularSwarm

agents = [Agent(agent_name="A1", model_name="gpt-4.1"), Agent(agent_name="A2", model_name="gpt-4.1")]
swarm = CircularSwarm(agents=agents, output_type="list")
print(swarm.run(["Task 1", "Task 2"]))
    ```

    ## What this demonstrates

    - Correct import and initialization flow for `various_alt_swarms`
    - Minimal execution path suitable for first integration tests
    - A baseline pattern to adapt for production use

    ## Related

    - Struct reference: [`swarms/structs/various_alt_swarms.md`](../structs/various_alt_swarms.md)
