# AOP Server Setup Example

This example demonstrates how to set up an Agent Orchestration Protocol (AOP) server with multiple specialized agents.

## Overview

The AOP server allows you to deploy multiple agents that can be discovered and called by other agents or clients in the network. This example shows how to create a server with specialized agents for different tasks.

## Code Example

```python
from swarms import Agent
from swarms.structs.aop import (
    AOP,
)

# Create specialized agents
research_agent = Agent(
    agent_name="Research-Agent",
    agent_description="Expert in research, data collection, and information gathering",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops=1,
    top_p=None,
    dynamic_temperature_enabled=True,
    system_prompt="""You are a research specialist. Your role is to:
    1. Gather comprehensive information on any given topic
    2. Analyze data from multiple sources
    3. Provide well-structured research findings
    4. Cite sources and maintain accuracy
    5. Present findings in a clear, organized manner
    
    Always provide detailed, factual information with proper context.""",
)

analysis_agent = Agent(
    agent_name="Analysis-Agent",
    agent_description="Expert in data analysis, pattern recognition, and generating insights",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops=1,
    top_p=None,
    dynamic_temperature_enabled=True,
    system_prompt="""You are an analysis specialist. Your role is to:
    1. Analyze data and identify patterns
    2. Generate actionable insights
    3. Create visualizations and summaries
    4. Provide statistical analysis
    5. Make data-driven recommendations
    
    Focus on extracting meaningful insights from information.""",
)

writing_agent = Agent(
    agent_name="Writing-Agent",
    agent_description="Expert in content creation, editing, and communication",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops=1,
    top_p=None,
    dynamic_temperature_enabled=True,
    system_prompt="""You are a writing specialist. Your role is to:
    1. Create engaging, well-structured content
    2. Edit and improve existing text
    3. Adapt tone and style for different audiences
    4. Ensure clarity and coherence
    5. Follow best practices in writing
    
    Always produce high-quality, professional content.""",
)

code_agent = Agent(
    agent_name="Code-Agent",
    agent_description="Expert in programming, code review, and software development",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops=1,
    top_p=None,
    dynamic_temperature_enabled=True,
    system_prompt="""You are a coding specialist. Your role is to:
    1. Write clean, efficient code
    2. Debug and fix issues
    3. Review and optimize code
    4. Explain programming concepts
    5. Follow best practices and standards
    
    Always provide working, well-documented code.""",
)

financial_agent = Agent(
    agent_name="Financial-Agent",
    agent_description="Expert in financial analysis, market research, and investment insights",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops=1,
    top_p=None,
    dynamic_temperature_enabled=True,
    system_prompt="""You are a financial specialist. Your role is to:
    1. Analyze financial data and markets
    2. Provide investment insights
    3. Assess risk and opportunities
    4. Create financial reports
    5. Explain complex financial concepts
    
    Always provide accurate, well-reasoned financial analysis.""",
)

# Basic usage - individual agent addition
deployer = AOP("MyAgentServer", verbose=True, port=5932)

agents = [
    research_agent,
    analysis_agent,
    writing_agent,
    code_agent,
    financial_agent,
]

deployer.add_agents_batch(agents)

deployer.run()
```

## Key Components

### 1. Agent Creation

Each agent is created with:

- **agent_name**: Unique identifier for the agent
- **agent_description**: Brief description of the agent's capabilities
- **model_name**: The language model to use
- **system_prompt**: Detailed instructions defining the agent's role and behavior

### 2. AOP Server Setup

- **Server Name**: "MyAgentServer" - identifies your server
- **Port**: 5932 - the port where the server will run
- **Verbose**: True - enables detailed logging

### 3. Agent Registration

- **add_agents_batch()**: Registers multiple agents at once
- Agents become available for discovery and remote calls

## Usage

1. **Start the Server**: Run the script to start the AOP server
2. **Agent Discovery**: Other agents or clients can discover available agents
3. **Remote Calls**: Agents can be called remotely by their names

## Server Features

- **Agent Discovery**: Automatically registers agents for network discovery
- **Remote Execution**: Agents can be called from other network nodes
- **Load Balancing**: Distributes requests across available agents
- **Health Monitoring**: Tracks agent status and availability

## Configuration Options

- **Port**: Change the port number as needed
- **Verbose**: Set to False for reduced logging
- **Server Name**: Use a descriptive name for your server
- **Authentication**: Add `auth_callback` to enable security (see below)

## Adding Authentication

You can secure your AOP server by adding a custom authentication callback:

### Simple API Key Authentication

```python
from swarms import Agent
from swarms.structs.aop import AOP

# Define authentication callback
def my_auth(token: str) -> bool:
    """Validate API keys."""
    valid_keys = {"api-key-1", "api-key-2", "api-key-3"}
    return token in valid_keys

# Create agents (same as above)
research_agent = Agent(
    agent_name="Research-Agent",
    model_name="claude-sonnet-4-5-20250929",
    max_loops=1,
    system_prompt="You are a research specialist.",
    temperature=0.7,
    top_p=None,
)

# Create AOP with authentication
deployer = AOP(
    server_name="SecureAgentServer",
    port=5932,
    verbose=True,
    auth_callback=my_auth,  # Enable authentication
)

deployer.add_agent(research_agent)
deployer.run()
```

### JWT Token Authentication

```python
import jwt

def jwt_auth(token: str) -> bool:
    """Validate JWT tokens."""
    try:
        payload = jwt.decode(token, "your-secret-key", algorithms=["HS256"])
        return payload.get("authorized", False)
    except:
        return False

deployer = AOP(
    server_name="JWT-SecureServer",
    port=5932,
    auth_callback=jwt_auth,
)
```

### Environment-Based Authentication

```python
import os

def env_auth(token: str) -> bool:
    """Validate tokens from environment."""
    valid_tokens = set(os.getenv("VALID_API_KEYS", "").split(","))
    return token in valid_tokens

deployer = AOP(
    server_name="Env-AuthServer",
    port=5932,
    auth_callback=env_auth,
)
```

### Client Usage with Authentication

When calling tools on an authenticated server:

```python
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def call_tool():
    url = "http://localhost:5932/mcp"

    async with streamablehttp_client(url) as ctx:
        read, write = ctx if len(ctx) == 2 else (ctx[0], ctx[1])

        async with ClientSession(read, write) as session:
            await session.initialize()

            # Include auth_token parameter
            result = await session.call_tool(
                name="Research-Agent",
                arguments={
                    "task": "Research AI trends",
                    "auth_token": "api-key-1"  # Required!
                },
            )

            print(result)

asyncio.run(call_tool())
```

### Authentication Rules

- If `auth_callback` is provided → authentication is enabled
- If `auth_callback` is None → no authentication required
- The callback function determines ALL security logic
- Return `True` to allow access, `False` to deny
- Failed authentication returns: `{"success": false, "error": "Authentication failed"}`

## Next Steps

- See [AOP Cluster Example](aop_cluster_example.md) for multi-server setups
- Check [AOP Reference](../structs/aop.md) for advanced configuration options
- Explore agent communication patterns in the examples directory
