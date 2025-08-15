#!/usr/bin/env python3
"""
Simple Working MCP Streaming Example
This demonstrates the core MCP streaming functionality working correctly.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def demonstrate_basic_functionality():
    """Demonstrate basic MCP streaming functionality."""
    print("Simple Working MCP Streaming Example")
    print("=" * 50)
    
    try:
        from swarms.structs import Agent
        from swarms.tools.mcp_unified_client import MCP_STREAMING_AVAILABLE
        
        print(f"MCP Streaming Available: {MCP_STREAMING_AVAILABLE}")
        
        # Create a simple agent with MCP streaming enabled
        agent = Agent(
            model_name="gpt-4o-mini",
            mcp_streaming_enabled=True,
            mcp_streaming_timeout=30,
            verbose=True
        )
        
        print("Agent created successfully with MCP streaming")
        
        # Test streaming status
        status = agent.get_mcp_streaming_status()
        print(f"Streaming Status: {status}")
        
        # Test enabling/disabling streaming at runtime
        print("\nTesting runtime streaming control...")
        
        # Disable streaming
        agent.disable_mcp_streaming()
        status = agent.get_mcp_streaming_status()
        print(f"   After disable: streaming_enabled = {status['streaming_enabled']}")
        
        # Enable streaming with custom callback
        def streaming_callback(chunk: str):
            print(f"   [STREAM] {chunk}", end="", flush=True)
        
        agent.enable_mcp_streaming(timeout=60, callback=streaming_callback)
        status = agent.get_mcp_streaming_status()
        print(f"   After enable: streaming_enabled = {status['streaming_enabled']}")
        print(f"   Has callback: {status['has_callback']}")
        
        # Test a simple task (without MCP tools for now)
        print("\nTesting simple agent task...")
        response = agent.run("Hello! Please introduce yourself briefly.")
        print(f"\nAgent response: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

def demonstrate_mcp_client():
    """Demonstrate MCP unified client functionality."""
    print("\nTesting MCP Unified Client...")
    
    try:
        from swarms.tools.mcp_unified_client import (
            MCPUnifiedClient,
            UnifiedTransportConfig,
            create_auto_config
        )
        
        # Test different transport configurations
        configs = [
            ("Auto HTTP", create_auto_config("http://localhost:8000/mcp")),
            ("STDIO", create_auto_config("stdio://python examples/mcp/swarms_api_mcp_server.py")),
            ("Streamable HTTP", create_auto_config("http://localhost:8001/mcp"))
        ]
        
        for name, config in configs:
            try:
                client = MCPUnifiedClient(config)
                transport_type = client._get_effective_transport()
                print(f"{name}: {transport_type}")
            except Exception as e:
                print(f"{name}: {str(e)[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"MCP client error: {e}")
        return False

def demonstrate_schemas():
    """Demonstrate MCP schemas functionality."""
    print("\nTesting MCP Schemas...")
    
    try:
        from swarms.schemas.mcp_schemas import (
            MCPConnection,
            MCPStreamingConfig,
            UnifiedTransportConfig
        )
        
        # Test MCP connection
        connection = MCPConnection(
            url="http://localhost:8000/mcp",
            transport="streamable_http",
            enable_streaming=True,
            timeout=30
        )
        print(f"MCP Connection: {connection.url} ({connection.transport})")
        
        # Test streaming config
        streaming_config = MCPStreamingConfig(
            enable_streaming=True,
            streaming_timeout=60,
            buffer_size=2048
        )
        print(f"Streaming Config: timeout={streaming_config.streaming_timeout}s")
        
        # Test unified transport config
        unified_config = UnifiedTransportConfig(
            transport_type="auto",
            url="http://localhost:8000/mcp",
            enable_streaming=True,
            auto_detect=True
        )
        print(f"Unified Config: {unified_config.transport_type}")
        
        return True
        
    except Exception as e:
        print(f"Schemas error: {e}")
        return False

def show_usage_examples():
    """Show usage examples for the MCP streaming functionality."""
    print("\nUsage Examples")
    print("=" * 50)
    
    print("\n1. **Basic Agent with MCP Streaming:**")
    print("""
from swarms.structs import Agent

agent = Agent(
    model_name="gpt-4o-mini",
    mcp_streaming_enabled=True,
    mcp_streaming_timeout=60,
    verbose=True
)

response = agent.run("Your task here")
""")
    
    print("\n2. **Agent with MCP Server:**")
    print("""
agent = Agent(
    model_name="gpt-4o-mini",
    mcp_url="http://localhost:8000/mcp",
    mcp_streaming_enabled=True,
    verbose=True
)

# The agent will automatically use MCP tools when available
response = agent.run("Use MCP tools to analyze this data")
""")
    
    print("\n3. **Runtime Streaming Control:**")
    print("""
# Enable streaming with custom callback
def my_callback(chunk: str):
    print(f"Streaming: {chunk}")

agent.enable_mcp_streaming(timeout=60, callback=my_callback)

# Check streaming status
status = agent.get_mcp_streaming_status()
print(f"Streaming enabled: {status['streaming_enabled']}")

# Disable streaming
agent.disable_mcp_streaming()
""")
    
    print("\n4. **MCP Unified Client:**")
    print("""
from swarms.tools.mcp_unified_client import (
    MCPUnifiedClient,
    create_auto_config
)

config = create_auto_config("http://localhost:8000/mcp")
client = MCPUnifiedClient(config)

# Get available tools
tools = client.get_tools_sync()

# Call a tool with streaming
results = client.call_tool_streaming_sync("tool_name", {"arg": "value"})
""")

def main():
    """Run the demonstration."""
    print("MCP Streaming Core Functionality Demo")
    print("=" * 60)
    
    # Run demonstrations
    demonstrations = [
        demonstrate_basic_functionality,
        demonstrate_mcp_client,
        demonstrate_schemas
    ]
    
    passed = 0
    total = len(demonstrations)
    
    for demo in demonstrations:
        if demo():
            passed += 1
        print()
    
    # Show usage examples
    show_usage_examples()
    
    print("\n" + "=" * 60)
    print(f"Demo Results: {passed}/{total} demonstrations successful")
    
    if passed == total:
        print("All demonstrations successful!")
        print("\nCore MCP streaming functionality is working correctly!")
        print("\nNext Steps:")
        print("   1. Set up an MCP server (e.g., examples/mcp/swarms_api_mcp_server.py)")
        print("   2. Configure your API keys (SWARMS_API_KEY)")
        print("   3. Start using MCP streaming in your applications!")
        print("\nResources:")
        print("   - working_swarms_api_mcp_demo.py")
        print("   - examples/mcp/ directory")
        print("   - PR_STREAMING_MCP_INTEGRATION.txt")
    else:
        print("Some demonstrations failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
