# Multi MCP Execution Example

This example demonstrates using a list of MCP servers with an `Agent`.

```python
import os
from swarms import Agent

# Configure multiple MCP URLs
os.environ["MCP_URLS"] = "http://localhost:8000/sse,http://localhost:9001/sse"

agent = Agent(
    agent_name="Multi-MCP-Agent",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Example payloads produced by your model
payloads = [
    {"function_name": "get_weather", "server_url": "http://localhost:8000/sse", "payload": {"city": "London"}},
    {"function_name": "get_news", "server_url": "http://localhost:9001/sse", "payload": {"topic": "ai"}},
]

agent.handle_multiple_mcp_tools(agent.mcp_urls, payloads)
```
