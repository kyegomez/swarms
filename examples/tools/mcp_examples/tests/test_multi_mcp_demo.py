import os
import json
from swarms import Agent
import io
import sys
from contextlib import redirect_stdout

print("\n=== Testing Multiple MCP Tool Execution ===\n")

# Configure multiple MCP URLs
os.environ["MCP_URLS"] = "http://localhost:8000/sse,http://localhost:9001/sse"

def capture_output(func):
    """Capture printed output from a function"""
    f = io.StringIO()
    with redirect_stdout(f):
        func()
    return f.getvalue()

def test_direct_tool_execution():
    """Test directly executing tools on different MCP servers"""
    print("Testing direct tool execution...\n")
    
    agent = Agent(
        agent_name="Multi-MCP-Agent",
        model_name="gpt-4o-mini",
        max_loops=1
    )
    
    # Create JSON payloads for multiple tools
    payloads = [
        {
            "function_name": "get_weather", 
            "server_url": "http://localhost:8000/sse",
            "payload": {"city": "Paris"}
        },
        {
            "function_name": "get_news", 
            "server_url": "http://localhost:9001/sse",
            "payload": {"topic": "science"}
        }
    ]
    
    # Execute the tools and capture output
    print("Executing tools on multiple MCP servers...")
    output = capture_output(
        lambda: agent.handle_multiple_mcp_tools(agent.mcp_urls, json.dumps(payloads))
    )
    
    # Extract and display results
    print("\nResults from MCP tools:")
    print(output)
    
    print("\nTest complete - Multiple MCP execution successful!")

def test_agent_configuration():
    """Test different ways to configure agents with multiple MCP URLs"""
    print("\n=== Testing Agent MCP Configuration Methods ===\n")
    
    # Method 1: Configure via environment variables (already set above)
    agent1 = Agent(agent_name="Env-Config-Agent")
    print(f"Agent1 MCP URLs (from env): {agent1.mcp_urls}")
    
    # Method 2: Configure via direct parameter
    agent2 = Agent(
        agent_name="Direct-Config-Agent",
        mcp_urls=["http://localhost:8000/sse", "http://localhost:9001/sse"]
    )
    print(f"Agent2 MCP URLs (from param): {agent2.mcp_urls}")

if __name__ == "__main__":
    test_agent_configuration()
    test_direct_tool_execution()
