import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import (
    streamablehttp_client as http_client,
)


async def create_agent_via_mcp():
    """Create and use an agent through MCP using streamable HTTP."""

    print("🔧 Starting MCP client connection...")

    # Connect to the MCP server using streamable HTTP
    try:
        async with http_client("http://localhost:8001/mcp") as (
            read,
            write,
            _,
        ):

            async with ClientSession(read, write) as session:
                try:
                    await session.initialize()
                    print("Session initialized successfully!")
                except Exception as e:
                    print(f"Session initialization failed: {e}")
                    raise

                # List available tools
                print("Listing available tools...")
                try:
                    tools = await session.list_tools()
                    print(
                        f"📋 Available tools: {[tool.name for tool in tools.tools]}"
                    )

                except Exception as e:
                    print(f"Failed to list tools: {e}")
                    raise

                # Create an agent using your tool
                print("Calling create_agent tool...")
                try:
                    arguments = {
                        "agent_name": "tech_expert",
                        "system_prompt": "You are a technology expert. Provide clear explanations.",
                        "model_name": "gpt-4",
                        "task": "Explain blockchain technology in simple terms",
                    }

                    result = await session.call_tool(
                        name="create_agent", arguments=arguments
                    )

                    # Result Handling
                    if hasattr(result, "content") and result.content:
                        if isinstance(result.content, list):
                            for content_item in result.content:
                                if hasattr(content_item, "text"):
                                    print(content_item.text)
                                else:
                                    print(content_item)
                        else:
                            print(result.content)
                    else:
                        print("No content returned from agent")

                    return result

                except Exception as e:
                    print(f"Tool call failed: {e}")
                    import traceback

                    traceback.print_exc()
                    raise

    except Exception as e:
        print(f"Connection failed: {e}")
        import traceback

        traceback.print_exc()
        raise


# Run the client
if __name__ == "__main__":
    asyncio.run(create_agent_via_mcp())
