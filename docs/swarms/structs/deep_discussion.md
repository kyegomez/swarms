# Deep Discussion

    Reference documentation for `swarms.structs.deep_discussion`.

    ## Overview

    This module provides production utilities for `deep discussion` in Swarms.

    ## Module Path

    ```python
    from swarms.structs.deep_discussion import ...
    ```

    ## Public API

    - `one_on_one_debate(max_loops, task, agents, img, output_type)`

    ## Quick Start

    ```python
    from swarms import Agent
from swarms.structs.deep_discussion import one_on_one_debate

a1 = Agent(agent_name="Architect", model_name="gpt-4.1")
a2 = Agent(agent_name="Reviewer", model_name="gpt-4.1")
print(one_on_one_debate(max_loops=2, task="Choose a queue system", agents=[a1, a2]))
    ```

    ## Tutorial

    See the runnable tutorial: [`swarms/examples/deep_discussion_example.md`](../examples/deep_discussion_example.md)

    ## Operational Notes

    - Validate credentials and model access before running LLM-backed examples.
    - Start with small inputs/tasks, then scale once behavior is verified.
