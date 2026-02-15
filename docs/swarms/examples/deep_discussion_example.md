# Deep Discussion Tutorial

    End-to-end usage for `deep_discussion`.

    ## Prerequisites

    - Python 3.10+
    - `pip install -U swarms`
    - Provider credentials configured when using hosted LLMs

    ## Example

    ```python
    from swarms import Agent
from swarms.structs.deep_discussion import one_on_one_debate

pro = Agent(agent_name="Pro", system_prompt="Argue in favor.", model_name="gpt-4.1", max_loops=1)
con = Agent(agent_name="Con", system_prompt="Argue against.", model_name="gpt-4.1", max_loops=1)

history = one_on_one_debate(
    max_loops=3,
    task="Should startups prioritize growth over profitability?",
    agents=[pro, con],
)
print(history)
    ```

    ## What this demonstrates

    - Basic construction/import pattern for `deep_discussion`
    - Minimal execution path you can adapt in production
    - Safe starting defaults for iteration

    ## Related

    - Struct reference: [`swarms/structs/deep_discussion.md`](../structs/deep_discussion.md)
