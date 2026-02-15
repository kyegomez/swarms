# Csv To Agent

    `csv_to_agent` reference documentation.

    **Module Path**: `swarms.structs.csv_to_agent`

    ## Overview

    Typed loader/validator for creating agents from CSV, JSON, or YAML configuration files.

    ## Public API

    - **`ModelName`**: `get_model_names()`, `is_valid_model()`
- **`FileType`**: No public methods documented in this module.
- **`AgentConfigDict`**: No public methods documented in this module.
- **`AgentValidationError`**: No public methods documented in this module.
- **`AgentValidator`**: `validate_config()`
- **`CSVAgentLoader`**: `file_type()`, `create_agent_file()`, `load_agents()`

    ## Quickstart

    ```python
    from swarms.structs.csv_to_agent import ModelName, FileType, AgentConfigDict
    ```

    ## Tutorial

    A runnable tutorial is available at [`swarms/examples/csv_to_agent_example.md`](../examples/csv_to_agent_example.md).

    ## Notes

    - Keep task payloads small for first runs.
    - Prefer deterministic prompts when comparing outputs across agents.
    - Validate provider credentials (for LLM-backed examples) before production use.
