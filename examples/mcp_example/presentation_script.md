
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

## 3. Demo Flow

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

## 4. Integration Highlights

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

## 5. Code Architecture

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

## 6. Key Benefits

1. Modular Architecture
2. Specialized Agents
3. Clean API Integration
4. Scalable Design
