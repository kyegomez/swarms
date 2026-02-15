# Deep Discussion Tutorial

    End-to-end tutorial for `swarms.structs.deep_discussion`.

    ## Prerequisites

    - Python 3.10+
    - `pip install -U swarms`

    ## Example

    ```python
    from swarms import Agent
from swarms.structs.deep_discussion import one_on_one_debate

a1 = Agent(agent_name="Architect", model_name="gpt-4.1")
a2 = Agent(agent_name="Reviewer", model_name="gpt-4.1")
print(one_on_one_debate(max_loops=2, task="Choose a queue system", agents=[a1, a2]))
    ```

    ## What this demonstrates

    - Correct import and initialization flow for `deep_discussion`
    - Minimal execution path suitable for first integration tests
    - A baseline pattern to adapt for production use

    ## Related

    - Struct reference: [`swarms/structs/deep_discussion.md`](../structs/deep_discussion.md)
