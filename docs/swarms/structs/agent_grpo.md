# Agent Grpo (Detailed)

    Reference documentation for `swarms.structs.agent_grpo`.

    ## Overview

    This module provides production utilities for `agent grpo` in Swarms.

    ## Module Path

    ```python
    from swarms.structs.agent_grpo import ...
    ```

    ## Public API

    - `AgenticGRPO`: `sample`, `compute_group_baseline`, `compute_advantages`, `run`, `get_all`

    ## Quick Start

    ```python
    from swarms import Agent
from swarms.structs.agent_grpo import AgenticGRPO

solver = Agent(agent_name="Math-Solver", model_name="gpt-4.1", max_loops=1)
grpo = AgenticGRPO(name="math-grpo", description="Sample + score", agent=solver, n=4, correct_answers=["42"])
print(grpo.run(task="What is 40 + 2?"))
    ```

    ## Tutorial

    See the runnable tutorial: [`swarms/examples/agent_grpo_example.md`](../examples/agent_grpo_example.md)

    ## Operational Notes

    - Validate credentials and model access before running LLM-backed examples.
    - Start with small inputs/tasks, then scale once behavior is verified.
