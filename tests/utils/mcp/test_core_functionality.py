#!/usr/bin/env python3
"""
Test Core MCP Streaming Functionality
This script tests the basic MCP streaming integration to ensure everything works.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    
    try:
        # Test basic swarms imports
        from swarms.structs import Agent
        print("Agent import successful")
        
        # Test MCP streaming imports
        from swarms.tools.mcp_unified_client import (
            MCPUnifiedClient,
            UnifiedTransportConfig,
            call_tool_streaming_sync,
            MCP_STREAMING_AVAILABLE
        )
        print("MCP unified client imports successful")
        print(f"   MCP Streaming Available: {MCP_STREAMING_AVAILABLE}")
        
        # Test MCP schemas
        from swarms.schemas.mcp_schemas import MCPConnection
        print("MCP schemas import successful")
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def test_agent_creation():
    """Test that Agent can be created with MCP streaming parameters."""
    print("\nTesting Agent creation with MCP streaming...")
    
    try:
        from swarms.structs import Agent
        
        # Test basic agent creation
        agent = Agent(
            model_name="gpt-4o-mini",
            mcp_streaming_enabled=True,
            mcp_streaming_timeout=30,
            verbose=True
        )
        print("Basic agent creation successful")
        
        # Test agent with MCP URL
        agent_with_mcp = Agent(
            model_name="gpt-4o-mini",
            mcp_url="http://localhost:8000/mcp",
            mcp_streaming_enabled=True,
            verbose=True
        )
        print("Agent with MCP URL creation successful")
        
        # Test streaming status
        status = agent_with_mcp.get_mcp_streaming_status()
        print(f"   Streaming status: {status}")
        
        return True
        
    except Exception as e:
        print(f"Agent creation error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

def test_mcp_client():
    """Test MCP unified client functionality."""
    print("\nTesting MCP unified client...")
    
    try:
        from swarms.tools.mcp_unified_client import (
            MCPUnifiedClient,
            UnifiedTransportConfig,
            create_auto_config
        )
        
        # Test config creation
        config = create_auto_config("http://localhost:8000/mcp")
        print("Auto config creation successful")
        
        # Test client creation
        client = MCPUnifiedClient(config)
        print("MCP client creation successful")
        
        # Test config validation
        print(f"   Transport type: {client._get_effective_transport()}")
        
        return True
        
    except Exception as e:
        print(f"MCP client error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

def test_streaming_functions():
    """Test streaming function availability."""
    print("\nTesting streaming functions...")
    
    try:
        from swarms.tools.mcp_unified_client import (
            call_tool_streaming_sync,
            execute_tool_call_streaming_unified
        )
        print("Streaming functions import successful")
        
        # Test function signatures
        import inspect
        sig = inspect.signature(call_tool_streaming_sync)
        print(f"   call_tool_streaming_sync signature: {sig}")
        
        return True
        
    except Exception as e:
        print(f"Streaming functions error: {e}")
        return False

def test_schemas():
    """Test MCP schemas functionality."""
    print("\nTesting MCP schemas...")
    
    try:
        from swarms.schemas.mcp_schemas import (
            MCPConnection,
            MCPStreamingConfig,
            UnifiedTransportConfig
        )
        print("MCP schemas import successful")
        
        # Test schema creation
        connection = MCPConnection(
            url="http://localhost:8000/mcp",
            transport="streamable_http",
            enable_streaming=True
        )
        print("MCP connection schema creation successful")
        
        streaming_config = MCPStreamingConfig(
            enable_streaming=True,
            streaming_timeout=30
        )
        print("MCP streaming config creation successful")
        
        return True
        
    except Exception as e:
        print(f"MCP schemas error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Core MCP Streaming Functionality")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_agent_creation,
        test_mcp_client,
        test_streaming_functions,
        test_schemas
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! Core functionality is working correctly.")
        print("\nWhat's working:")
        print("   - MCP streaming imports")
        print("   - Agent creation with MCP parameters")
        print("   - MCP unified client")
        print("   - Streaming functions")
        print("   - MCP schemas")
        print("\nYou can now use MCP streaming functionality!")
    else:
        print("Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("   - Install required dependencies: pip install mcp mcp[streamable-http] httpx")
        print("   - Check that all files are in the correct locations")
        print("   - Verify that imports are working correctly")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
