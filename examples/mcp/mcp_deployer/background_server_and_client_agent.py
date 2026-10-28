import socket

from swarms import Agent, MCPDeployer
from swarms.schemas.mcp_schemas import MCPConnection


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


poet = Agent(
    agent_name="Poet",
    agent_description="Writes a two-line poem about the topic it is given.",
    system_prompt="Write exactly two short lines of poetry about the topic. Nothing else.",
    model_name="claude-haiku-4-5-20251001",
    max_loops=1,
    print_on=False,
)

if __name__ == "__main__":
    with MCPDeployer(
        poet, api_keys=["demo-key"], port=free_port()
    ) as server:
        print(f"serving {server.tool_name} at {server.url}")

        client = Agent(
            agent_name="Client",
            system_prompt="Use the poet tool to get a poem, then return it verbatim.",
            model_name="claude-haiku-4-5-20251001",
            mcp_url=MCPConnection(url=server.url, api_key="demo-key"),
            max_loops=2,
            print_on=False,
            output_type="final",
        )
        client.run("Get a poem about lighthouses.")

        print("\nwhat the served agent produced:")
        print(poet.short_memory.get_final_message_content())
    # Leaving the `with` block stops the server.
