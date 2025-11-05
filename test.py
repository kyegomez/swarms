"""
Simple AOP Client with Authentication

Just pass your token when calling tools. That's it.
The server's auth_callback determines if it's valid.
"""

import json
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def call_server():
    """Call the AOP server with authentication."""

    url = "http://localhost:5932/mcp"

    async with streamablehttp_client(url, timeout=10) as ctx:
        if len(ctx) == 2:
            read, write = ctx
        else:
            read, write, *_ = ctx

        async with ClientSession(read, write) as session:
            await session.initialize()

            print("\n" + "=" * 60)
            print("Calling discover_agents...")
            print("=" * 60 + "\n")

            result = await session.call_tool(
                name="discover_agents",
                arguments={
                    "auth_token": "mytoken1234" 
                },
            )

            print(json.dumps(result.model_dump(), indent=2))

            print("\n" + "=" * 60)
            print("Calling Research-Agent...")
            print("=" * 60 + "\n")

            result = await session.call_tool(
                name="Research-Agent",
                arguments={
                    "task": "What is Python?",
                    "auth_token": "mytoken123" 
                },
            )

            print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    print("\n🔐 Simple Auth Client")
    print("Token: mytoken123\n")
    asyncio.run(call_server())
