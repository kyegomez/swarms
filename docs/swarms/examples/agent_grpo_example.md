# Agent Grpo Tutorial

    End-to-end usage for `agent_grpo`.

    ## Prerequisites

    - Python 3.10+
    - `pip install -U swarms`
    - Provider credentials configured when using hosted LLMs

    ## Example

    ```python
    from swarms import Agent
from swarms.structs.agent_grpo import AgenticGRPO

solver = Agent(
    agent_name="Math-Solver",
    system_prompt="Solve math problems with concise reasoning.",
    model_name="gpt-4.1",
    max_loops=1,
)

grpo = AgenticGRPO(
    name="math-grpo",
    description="Sample and score candidate completions",
    agent=solver,
    n=4,
    correct_answers=["42"],
)

correct = grpo.run(task="What is 40 + 2?")
print(correct)
print("baseline:", grpo.get_group_baseline())
    ```

    ## What this demonstrates

    - Basic construction/import pattern for `agent_grpo`
    - Minimal execution path you can adapt in production
    - Safe starting defaults for iteration

    ## Related

    - Struct reference: [`swarms/structs/agent_grpo.md`](../structs/agent_grpo.md)
