# Basic Agent Setup with MCP

## Overview

This document shows how to set up a basic Swarms agent with MCP (Model Context Protocol) integration for client-side operations.

## Basic Agent Setup

```python
from swarms import Agent
from swarms.tools.mcp_integration import MCPServerSseParams

# Configure MCP server parameters
mcp_params = MCPServerSseParams(
    url="http://localhost:8081/sse",  # MCP server SSE endpoint
    headers={"Accept": "text/event-stream"},  # Required for SSE
    timeout=5.0  # Connection timeout in seconds
)

# Initialize agent with MCP configuration
agent = Agent(
    agent_name="basic_agent",  # Name of your agent
    system_prompt="You are a helpful assistant",  # Agent's system prompt
    mcp_servers=[mcp_params],  # List of MCP server configurations
    max_loops=5,  # Maximum number of loops for task execution
    verbose=True  # Enable verbose output
)

# Run the agent
result = agent.run("Your task here")
print(result)
```

## Required Parameters

1. **MCP Server Parameters**:
   - `url`: The SSE endpoint of your MCP server
   - `headers`: Must include `Accept: text/event-stream`
   - `timeout`: Connection timeout in seconds

2. **Agent Parameters**:
   - `agent_name`: Name of your agent
   - `system_prompt`: Agent's system prompt
   - `mcp_servers`: List of MCP server configurations
   - `max_loops`: Maximum number of loops for task execution
   - `verbose`: Enable verbose output for debugging

## Example Usage

```python
# Create agent
agent = Agent(
    agent_name="math_agent",
    system_prompt="You are a math assistant",
    mcp_servers=[mcp_params],
    max_loops=5,
    verbose=True
)

# Run a math task
result = agent.run("Add 5 and 3")
print(result)  # Should return 8
``` 