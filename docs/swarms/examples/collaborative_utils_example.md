# Collaborative Utils Tutorial

    End-to-end usage for `collaborative_utils`.

    ## Prerequisites

    - Python 3.10+
    - `pip install -U swarms`
    - Provider credentials configured when using hosted LLMs

    ## Example

    ```python
    from swarms import Agent
from swarms.structs.collaborative_utils import talk_to_agent

alice = Agent(agent_name="Alice", system_prompt="You are analytical.", model_name="gpt-4.1", max_loops=1)
bob = Agent(agent_name="Bob", system_prompt="You are critical.", model_name="gpt-4.1", max_loops=1)

conversation = talk_to_agent(
    current_agent=alice,
    agents=[alice, bob],
    task="Evaluate this launch plan.",
    agent_name="Bob",
    max_loops=2,
)
print(conversation)
    ```

    ## What this demonstrates

    - Basic construction/import pattern for `collaborative_utils`
    - Minimal execution path you can adapt in production
    - Safe starting defaults for iteration

    ## Related

    - Struct reference: [`swarms/structs/collaborative_utils.md`](../structs/collaborative_utils.md)
