# Collaborative Utils Tutorial

    End-to-end tutorial for `swarms.structs.collaborative_utils`.

    ## Prerequisites

    - Python 3.10+
    - `pip install -U swarms`

    ## Example

    ```python
    from swarms import Agent
from swarms.structs.collaborative_utils import talk_to_agent

alice = Agent(agent_name="alice", model_name="gpt-4.1")
bob = Agent(agent_name="bob", model_name="gpt-4.1")
print(talk_to_agent(current_agent=alice, agents=[alice, bob], task="Debate caching strategy", agent_name="bob", max_loops=2))
    ```

    ## What this demonstrates

    - Correct import and initialization flow for `collaborative_utils`
    - Minimal execution path suitable for first integration tests
    - A baseline pattern to adapt for production use

    ## Related

    - Struct reference: [`swarms/structs/collaborative_utils.md`](../structs/collaborative_utils.md)
