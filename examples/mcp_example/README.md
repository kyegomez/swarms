
# MCP Integration Documentation

## Overview
This implementation connects an agent with a Math Server using the Model Context Protocol (MCP). The system consists of:

1. Math Server (math_server.py) - Handles mathematical calculations
2. Test Integration (test_integration.py) - Client that connects to the math server

## Key Components

### Math Agent Setup
```python
math_server = MCPServerSseParams(
    url="http://0.0.0.0:6274",
    headers={"Content-Type": "application/json"},
)

math_agent = Agent(
    agent_name="Math-Agent",
    system_prompt="You are a math assistant. Process mathematical operations.",
    max_loops=1,
    mcp_servers=[math_server]
)
```

### Task Flow
1. User inputs a math operation
2. Math agent processes the request
3. Request is sent to math server via MCP
4. Result is returned through the agent

## Testing
Run the integration test:
```bash
python examples/mcp_example/math_server.py  # Start server
python examples/mcp_example/test_integration.py  # Run client
```

## Implementation Notes
- Server runs on port 6274
- Uses SSE (Server-Sent Events) for transport
- Handles basic math operations (add, subtract, multiply, divide)
