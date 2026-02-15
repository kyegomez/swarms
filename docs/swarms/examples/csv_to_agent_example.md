# CSV to Agent Tutorial

    End-to-end tutorial for `swarms.structs.csv_to_agent`.

    ## Prerequisites

    - Python 3.10+
    - `pip install -U swarms`

    ## Example

    ```python
    from swarms.structs.csv_to_agent import CSVAgentLoader

loader = CSVAgentLoader(file_path="./agents.yaml", max_workers=4)
# loader.create_agent_file([...])
agents = loader.load_agents()
print(f"Loaded {len(agents)} agents")
    ```

    ## What this demonstrates

    - Correct import and initialization flow for `csv_to_agent`
    - Minimal execution path suitable for first integration tests
    - A baseline pattern to adapt for production use

    ## Related

    - Struct reference: [`swarms/structs/csv_to_agent.md`](../structs/csv_to_agent.md)
