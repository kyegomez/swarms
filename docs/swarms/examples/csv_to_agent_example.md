# Csv To Agent Tutorial

    End-to-end usage for `csv_to_agent`.

    ## Prerequisites

    - Python 3.10+
    - `pip install -U swarms`
    - Provider credentials configured when using hosted LLMs

    ## Example

    ```python
    from swarms.structs.csv_to_agent import CSVAgentLoader

loader = CSVAgentLoader(file_path="./agents.yaml", max_workers=4)

# Create config file from dicts (or AgentSpec objects)
loader.create_agent_file([
    {
        "agent_name": "Analyst",
        "system_prompt": "Analyze financial statements.",
        "model_name": "gpt-4.1",
        "max_loops": 1,
    }
])

agents = loader.load_agents()
print([a.agent_name for a in agents])
    ```

    ## What this demonstrates

    - Basic construction/import pattern for `csv_to_agent`
    - Minimal execution path you can adapt in production
    - Safe starting defaults for iteration

    ## Related

    - Struct reference: [`swarms/structs/csv_to_agent.md`](../structs/csv_to_agent.md)
