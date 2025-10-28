import asyncio
import json

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
    async with streamablehttp_client(url, timeout=30) as ctx:
        if len(ctx) == 2:
            read, write = ctx
        else:
            read, write, *_ = ctx

        async with ClientSession(read, write) as session:
            await session.initialize()
            arguments = {"task": task}
            if img is not None:
                arguments["img"] = img
            if imgs is not None:
                arguments["imgs"] = imgs
            if correct_answer is not None:
                arguments["correct_answer"] = correct_answer
            result = await session.call_tool(
                name=tool_name, arguments=arguments
            )
            return result.model_dump()


async def list_available_tools(url: str) -> dict:
    async with streamablehttp_client(url, timeout=30) as ctx:
        if len(ctx) == 2:
            read, write = ctx
        else:
            read, write, *_ = ctx

        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            return tools.model_dump()


def main():
    url = "http://localhost:5932/mcp"
    tool_name = "Research-Agent"
    task = "What are the latest experimental drug trials coming up in the next 6 months?"

    tools_info = asyncio.run(list_available_tools(url))
    print("Available tools:")
    print(json.dumps(tools_info, indent=2))

    ####### Step 2: Call the agent

    print(f"\nCalling tool '{tool_name}' with task...\n")
    result = asyncio.run(
        call_agent_tool_raw(url=url, tool_name=tool_name, task=task)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
