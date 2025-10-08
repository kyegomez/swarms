import json
import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def call_agent_tool_raw(
    url: str,
    tool_name: str,
    task: str,
    img: str | None = None,
    imgs: list[str] | None = None,
    correct_answer: str | None = None,
) -> dict:
    """
    Call a specific agent tool on an MCP server using the raw MCP client.

    Args:
        url: MCP server URL (e.g., "http://localhost:5932/mcp").
        tool_name: Name of the tool/agent to invoke.
        task: Task prompt to execute.
        img: Optional single image path/URL.
        imgs: Optional list of image paths/URLs.
        correct_answer: Optional expected answer for validation.

    Returns:
        A dict containing the tool's JSON response.
    """
    # Open a raw MCP client connection over streamable HTTP
    async with streamablehttp_client(url, timeout=30) as ctx:
        if len(ctx) == 2:
            read, write = ctx
        else:
            read, write, *_ = ctx

        async with ClientSession(read, write) as session:
            # Initialize the MCP session
            await session.initialize()

            # Prepare arguments in the canonical AOP tool format
            arguments: dict = {"task": task}
            if img is not None:
                arguments["img"] = img
            if imgs is not None:
                arguments["imgs"] = imgs
            if correct_answer is not None:
                arguments["correct_answer"] = correct_answer

            # Invoke the tool by name
            result = await session.call_tool(
                name=tool_name, arguments=arguments
            )

            # Convert to dict for return/printing
            return result.model_dump()


async def list_available_tools(url: str) -> dict:
    """
    List tools from an MCP server using the raw client.

    Args:
        url: MCP server URL (e.g., "http://localhost:5932/mcp").

    Returns:
        A dict representation of the tools listing.
    """
    async with streamablehttp_client(url, timeout=30) as ctx:
        if len(ctx) == 2:
            read, write = ctx
        else:
            read, write, *_ = ctx

        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            return tools.model_dump()


def main() -> None:
    """
    Demonstration entrypoint: list tools, then call a specified tool with a task.
    """
    url = "http://localhost:5932/mcp"
    tool_name = "Research-Agent"  # Change to your agent tool name
    task = "Summarize the latest advances in agent orchestration protocols."

    # List tools
    tools_info = asyncio.run(list_available_tools(url))
    print("Available tools:")
    print(json.dumps(tools_info, indent=2))

    # Call the tool
    print(f"\nCalling tool '{tool_name}' with task...\n")
    result = asyncio.run(
        call_agent_tool_raw(
            url=url,
            tool_name=tool_name,
            task=task,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
