# CSV to Agent

    Reference documentation for `swarms.structs.csv_to_agent`.

    ## Overview

    This module provides production utilities for `csv to agent` in Swarms.

    ## Module Path

    ```python
    from swarms.structs.csv_to_agent import ...
    ```

    ## Public API

    - `CSVAgentLoader`: `create_agent_file`, `load_agents`; `AgentValidator.validate_config`

    ## Quick Start

    ```python
    from swarms.structs.csv_to_agent import CSVAgentLoader

loader = CSVAgentLoader(file_path="./agents.yaml", max_workers=4)
# loader.create_agent_file([...])
agents = loader.load_agents()
print(f"Loaded {len(agents)} agents")
    ```

    ## Tutorial

    See the runnable tutorial: [`swarms/examples/csv_to_agent_example.md`](../examples/csv_to_agent_example.md)

    ## Operational Notes

    - Validate credentials and model access before running LLM-backed examples.
    - Start with small inputs/tasks, then scale once behavior is verified.
- Validation checks model names against `litellm.model_list`; keep model names provider-compatible.
