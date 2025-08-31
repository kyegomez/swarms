# MCP Agent Tool Documentation

## Introduction to MCP and Agent Running

The Model Context Protocol (MCP) provides a standardized way to create and manage AI agents through a server-client architecture. Running agents on MCP offers several key benefits:

- **Standardized Interface**: Consistent API for agent creation and management across different systems
- **Scalability**: Handle multiple agents simultaneously through a single MCP server
- **Interoperability**: Agents can be called from any MCP-compatible client
- **Resource Management**: Centralized control over agent lifecycle and resources
- **Protocol Compliance**: Follows the established MCP standard for AI tool integration

## Step 1: Setup and Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Required Packages

Install the necessary packages using pip:

```bash
# Install MCP SDK and FastMCP
pip install mcp fastmcp

# Install Swarms framework
pip install swarms

# Install additional dependencies
pip install loguru
```

### Verify Installation

```python
# Test imports
from mcp.server.fastmcp import FastMCP
from swarms import Agent

print("MCP and Swarms installed successfully!")
```

## Step 2: MCP Server Setup

Create the MCP server file that will handle agent creation requests:

```python
from mcp.server.fastmcp import FastMCP
from swarms import Agent

mcp = FastMCP("MCPAgentTool")

@mcp.tool(
    name="create_agent",
    description="Create an agent with the specified name, system prompt, and model, then run a task.",
)
def create_agent(agent_name: str, system_prompt: str, model_name: str, task: str) -> str:
    """
    Create an agent with the given parameters and execute the specified task.

    Args:
        agent_name (str): The name of the agent to create.
        system_prompt (str): The system prompt to initialize the agent with.
        model_name (str): The model name to use for the agent.
        task (str): The task for the agent to perform.

    Returns:
        str: The result of the agent running the given task.
    """
    agent = Agent(
        agent_name=agent_name,
        system_prompt=system_prompt,
        model_name=model_name,
    )
    return agent.run(task)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

Save this as `mcp_agent_tool.py` and run it to start the MCP server:

```bash
python mcp_agent_tool.py
```

## Step 3: Basic Client-side Setup: Single Agent

Create a client file to interact with the MCP server and run a single agent:

```python
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
        async with http_client("http://localhost:8000/mcp") as (read, write, _):
            
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
                    print(f"📋 Available tools: {[tool.name for tool in tools.tools]}")
                        
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
                        "task": "Explain blockchain technology in simple terms"
                    }
                    
                    result = await session.call_tool(
                        name="create_agent",
                        arguments=arguments
                    )
                    
                    # Result Handling
                    if hasattr(result, 'content') and result.content:
                        if isinstance(result.content, list):
                            for content_item in result.content:
                                if hasattr(content_item, 'text'):
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
```

## Step 4: Advanced Client-side Setup: Multiple Agents

Create a multi-agent system that chains multiple agents together for complex workflows:

```python
import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import (
    streamablehttp_client as http_client,
)

async def create_agent_via_mcp(session, agent_name, system_prompt, model_name, task):
    """Create and use an agent through MCP using streamable HTTP."""
    print(f"🔧 Creating agent '{agent_name}' with task: {task}")
    try:
        arguments = {
            "agent_name": agent_name,
            "system_prompt": system_prompt,
            "model_name": model_name,
            "task": task
        }
        result = await session.call_tool(
            name="create_agent",
            arguments=arguments
        )
        # Result Handling
        output = None
        if hasattr(result, 'content') and result.content:
            if isinstance(result.content, list):
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
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
        async with http_client("http://localhost:8000/mcp") as (read, write, _):
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
                    print(f"📋 Available tools: {[tool.name for tool in tools.tools]}")
                except Exception as e:
                    print(f"Failed to list tools: {e}")
                    raise

                # Sequential Multi-Agent System
                # Agent 1: Tech Expert explains blockchain
                agent1_name = "tech_expert"
                agent1_prompt = "You are a technology expert. Provide clear explanations."
                agent1_model = "gpt-4"
                agent1_task = "Explain blockchain technology in simple terms"

                agent1_output = await create_agent_via_mcp(
                    session,
                    agent1_name,
                    agent1_prompt,
                    agent1_model,
                    agent1_task
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
                    agent2_task
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
                    agent3_task
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
```

## Summary: Complete Setup Steps for Agent Initialization and Setup on MCP

Here's a complete overview of all the steps needed to set up your agent initialization and setup on MCP:

### **Step-by-Step Summary:**

1. **📦 Package Installation** - Install MCP SDK, FastMCP, Swarms, and dependencies
2. **🔧 Server Creation** - Create the MCP server with agent creation tool
3. **🚀 Server Startup** - Run the MCP server to handle client requests
4. **📱 Basic Client** - Create a simple client to run single agents
5. **🔄 Advanced Client** - Build multi-agent workflows with sequential processing

### **What You'll Have After Following These Steps:**

- ✅ **MCP Server** running and ready to handle agent creation requests
- ✅ **Single Agent Client** for basic agent tasks
- ✅ **Multi-Agent Client** for complex, chained workflows
- ✅ **Complete System** for dynamic agent creation and management
- ✅ **Scalable Architecture** that can handle multiple concurrent agent requests

### **Key Benefits Achieved:**

- **Standardized Interface** for agent management
- **Scalable Architecture** for multiple agents
- **Protocol Compliance** with MCP standards
- **Resource Management** for efficient agent lifecycle
- **Interoperability** with any MCP-compatible client

This setup gives you a complete, production-ready system for running AI agents through the Model Context Protocol!

## Connect With Us

If you'd like technical support, join our Discord below and stay updated on our Twitter for new updates!

| Platform | Link | Description |
|----------|------|-------------|
| 📚 Documentation | [docs.swarms.world](https://docs.swarms.world) | Official documentation and guides |
| 📝 Blog | [Medium](https://medium.com/@kyeg) | Latest updates and technical articles |
| 💬 Discord | [Join Discord](https://discord.gg/EamjgSaEQf) | Live chat and community support |
| 🐦 Twitter | [@kyegomez](https://twitter.com/kyegomez) | Latest news and announcements |
| 👥 LinkedIn | [The Swarm Corporation](https://www.linkedin.com/company/the-swarm-corporation) | Professional network and updates |
| 📺 YouTube | [Swarms Channel](https://www.youtube.com/channel/UC9yXyitkbU_WSy7bd_41SqQ) | Tutorials and demos |
| 🎫 Events | [Sign up here](https://lu.ma/5p2jnc2v) | Join our community events |
