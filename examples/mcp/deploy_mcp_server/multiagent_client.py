import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import (
    streamablehttp_client as http_client,
)


async def create_agent_via_mcp(
    session, agent_name, system_prompt, model_name, task
):
    """Create and use an agent through MCP using streamable HTTP."""
    print(f"🔧 Creating agent '{agent_name}' with task: {task}")
    try:
        arguments = {
            "agent_name": agent_name,
            "system_prompt": system_prompt,
            "model_name": model_name,
            "task": task,
        }
        result = await session.call_tool(
            name="create_agent", arguments=arguments
        )
        # Result Handling
        output = None
        if hasattr(result, "content") and result.content:
            if isinstance(result.content, list):
                for content_item in result.content:
                    if hasattr(content_item, "text"):
                        print(content_item.text)
                        output = content_item.text
                    else:
                        print(content_item)
                        output = content_item
            else:
                print(result.content)
                output = result.content
        else:
            print("No content returned from agent")
        return output
    except Exception as e:
        print(f"Tool call failed: {e}")
        import traceback

        traceback.print_exc()
        raise


async def main():
    print("🔧 Starting MCP client connection...")

    try:
        async with http_client("http://localhost:8000/mcp") as (
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

                # Sequential Multi-Agent System
                # Agent 1: Tech Expert explains blockchain
                agent1_name = "tech_expert"
                agent1_prompt = "You are a technology expert. Provide clear explanations."
                agent1_model = "gpt-4"
                agent1_task = (
                    "Explain blockchain technology in simple terms"
                )

                agent1_output = await create_agent_via_mcp(
                    session,
                    agent1_name,
                    agent1_prompt,
                    agent1_model,
                    agent1_task,
                )

                # Agent 2: Legal Expert analyzes the explanation from Agent 1
                agent2_name = "legal_expert"
                agent2_prompt = "You are a legal expert. Analyze the following explanation for legal implications."
                agent2_model = "gpt-4"
                agent2_task = f"Analyze the following explanation for legal implications:\n\n{agent1_output}"

                agent2_output = await create_agent_via_mcp(
                    session,
                    agent2_name,
                    agent2_prompt,
                    agent2_model,
                    agent2_task,
                )

                # Agent 3: Educator simplifies the legal analysis for students
                agent3_name = "educator"
                agent3_prompt = "You are an educator. Summarize the following legal analysis in simple terms for students."
                agent3_model = "gpt-4"
                agent3_task = f"Summarize the following legal analysis in simple terms for students:\n\n{agent2_output}"

                agent3_output = await create_agent_via_mcp(
                    session,
                    agent3_name,
                    agent3_prompt,
                    agent3_model,
                    agent3_task,
                )

                print("\n=== Final Output from Educator Agent ===")
                print(agent3_output)

    except Exception as e:
        print(f"Connection failed: {e}")
        import traceback

        traceback.print_exc()
        raise


# Run the client
if __name__ == "__main__":
    asyncio.run(main())
