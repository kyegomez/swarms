
# MCP Integration Demo Script

## 1. Setup & Architecture Overview

```bash
# Terminal 1: Start Stock Server
python examples/mcp_example/mock_stock_server.py

# Terminal 2: Start Math Server
python examples/mcp_example/mock_math_server.py

# Terminal 3: Start Multi-Agent System
python examples/mcp_example/mock_multi_agent.py
```

## 2. Key Components

### Server-Side:
- FastMCP servers running on ports 8000 and 8001
- Math Server provides: add, multiply, divide operations
- Stock Server provides: price lookup, moving average calculations

### Client-Side:
- Multi-agent system with specialized agents
- MCPServerSseParams for server connections
- Automatic task routing based on agent specialization

## 3. MCP Integration Details

### Server Implementation:
```python
# Math Server Example
from fastmcp import FastMCP

mcp = FastMCP("Math-Server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b
```

### Client Integration:
```python
from swarms.tools.mcp_integration import MCPServerSseParams

# Configure MCP server connection
server = MCPServerSseParams(
    url="http://0.0.0.0:8000",
    headers={"Content-Type": "application/json"}
)

# Initialize agent with MCP capabilities
agent = Agent(
    agent_name="Calculator",
    mcp_servers=[server],
    max_loops=1
)
```

## 4. Demo Flow

1. Math Operations:
```
Enter a math problem: 5 plus 3
Enter a math problem: 10 times 4
```

2. Stock Analysis:
```
Enter a math problem: get price of AAPL
Enter a math problem: calculate moving average of [10,20,30,40,50] over 3 periods
```

## 5. Integration Highlights

1. Server Configuration:
- FastMCP initialization
- Tool registration using decorators
- SSE transport setup

2. Client Integration:
- MCPServerSseParams configuration
- Agent specialization
- Task routing logic

3. Communication Flow:
- Client request → Agent processing → MCP server → Response handling

4. Error Handling:
- Graceful error management
- Automatic retry mechanisms
- Clear error reporting

## 6. Code Architecture

### Server Example (Math Server):
```python
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b
```

### Client Example (Multi-Agent):
```python
calculator = MathAgent("Calculator", "http://0.0.0.0:8000")
stock_analyst = MathAgent("StockAnalyst", "http://0.0.0.0:8001")
```

## 7. Key Benefits

1. Modular Architecture
2. Specialized Agents
3. Clean API Integration
4. Scalable Design
5. Standardized Communication Protocol
6. Easy Tool Registration
7. Flexible Server Implementation

## 8. Testing & Validation

1. Basic Connectivity:
```python
def test_server_connection():
    params = {"url": "http://0.0.0.0:8000"}
    server = MCPServerSse(params)
    asyncio.run(server.connect())
    assert server.session is not None
```

2. Tool Execution:
```python
def test_tool_execution():
    params = {"url": "http://0.0.0.0:8000"}
    function_call = {
        "tool_name": "add",
        "arguments": {"a": 5, "b": 3}
    }
    result = mcp_flow(params, function_call)
    assert result is not None
```
