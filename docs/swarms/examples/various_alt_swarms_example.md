# Various Alt Swarms Tutorial

    End-to-end usage for `various_alt_swarms`.

    ## Prerequisites

    - Python 3.10+
    - `pip install -U swarms`
    - Provider credentials configured when using hosted LLMs

    ## Example

    ```python
    from swarms import Agent
from swarms.structs.various_alt_swarms import CircularSwarm, Broadcast

agents = [
    Agent(agent_name="A1", system_prompt="Analyze", model_name="gpt-4.1", max_loops=1),
    Agent(agent_name="A2", system_prompt="Summarize", model_name="gpt-4.1", max_loops=1),
]

circular = CircularSwarm(agents=agents, output_type="list")
print(circular.run(["Analyze product feedback", "Extract action items"]))

sender = agents[0]
receivers = [agents[1]]
broadcast = Broadcast(sender=sender, receivers=receivers)
print(broadcast.run("Announce Q2 goals"))
    ```

    ## What this demonstrates

    - Basic construction/import pattern for `various_alt_swarms`
    - Minimal execution path you can adapt in production
    - Safe starting defaults for iteration

    ## Related

    - Struct reference: [`swarms/structs/various_alt_swarms.md`](../structs/various_alt_swarms.md)
