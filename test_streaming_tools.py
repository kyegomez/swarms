#!/usr/bin/env python3
"""
Test script to reproduce and verify the fix for issue #936:
Agent tool usage fails when streaming is enabled.
"""

from typing import List
import json
from swarms import Agent
import os
import logging

# Set up logging to see the tool execution logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def simple_calculator_tool(operation: str, num1: float, num2: float) -> str:
    """
    Simple calculator tool for testing.
    
    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        num1: First number
        num2: Second number
    
    Returns:
        str: Result of the calculation
    """
    logger.info(f"Calculator tool called: {operation}({num1}, {num2})")
    
    if operation == "add":
        result = num1 + num2
    elif operation == "subtract":
        result = num1 - num2
    elif operation == "multiply":
        result = num1 * num2
    elif operation == "divide":
        if num2 == 0:
            return "Error: Division by zero"
        result = num1 / num2
    else:
        return f"Error: Unknown operation {operation}"
    
    return f"The result of {operation}({num1}, {num2}) is {result}"


def test_agent_streaming_with_tools():
    """Test that agent can use tools when streaming is enabled"""
    
    print("🧪 Testing Agent with Streaming + Tools...")
    
    # Create agent with streaming enabled
    agent = Agent(
        agent_name="Calculator-Agent",
        system_prompt="""
        You are a helpful calculator assistant. When asked to perform calculations,
        use the simple_calculator_tool to compute the result.
        
        Available tool:
        - simple_calculator_tool(operation, num1, num2): Performs basic calculations
        
        Always use the tool for calculations instead of doing them yourself.
        """,
        model_name="gpt-3.5-turbo",  # Using a common model for testing
        max_loops=2,  # Allow for tool execution + response
        verbose=True,
        streaming_on=True,  # THIS IS THE KEY - streaming enabled
        print_on=True,
        tools=[simple_calculator_tool],
        # Add any necessary API keys from environment
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    # Test task that should trigger tool usage
    task = "Please calculate 25 + 17 for me using the calculator tool"
    
    print(f"\n📝 Task: {task}")
    print("\n🔄 Running agent with streaming + tools...")
    
    try:
        result = agent.run(task)
        print(f"\n✅ Result: {result}")
        
        # Check if the tool was actually executed by looking at memory
        memory_history = agent.short_memory.return_history_as_string()
        
        if "Tool Executor" in memory_history:
            print("✅ SUCCESS: Tool execution found in memory history!")
            return True
        else:
            print("❌ FAILURE: No tool execution found in memory history")
            print("Memory history:")
            print(memory_history)
            return False
            
    except Exception as e:
        print(f"❌ FAILURE: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_streaming_without_tools():
    """Test that agent works normally when streaming is enabled but no tools needed"""
    
    print("\n🧪 Testing Agent with Streaming (No Tools)...")
    
    agent = Agent(
        agent_name="Simple-Agent",
        system_prompt="You are a helpful assistant.",
        model_name="gpt-3.5-turbo",
        max_loops=1,
        verbose=True,
        streaming_on=True,  # Streaming enabled
        print_on=True,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    task = "What is the capital of France?"
    
    print(f"\n📝 Task: {task}")
    print("\n🔄 Running agent with streaming (no tools)...")
    
    try:
        result = agent.run(task)
        print(f"\n✅ Result: {result}")
        
        if "Paris" in str(result):
            print("✅ SUCCESS: Agent responded correctly without tools")
            return True
        else:
            print("❌ FAILURE: Agent didn't provide expected response")
            return False
            
    except Exception as e:
        print(f"❌ FAILURE: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔧 Testing Fix for Issue #936: Agent Tool Usage with Streaming")
    print("=" * 60)
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. Tests may fail.")
        print("Please set OPENAI_API_KEY environment variable to run tests.")
    
    # Run tests
    test1_passed = test_agent_streaming_without_tools()
    test2_passed = test_agent_streaming_with_tools()
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    print(f"  Test 1 (Streaming without tools): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  Test 2 (Streaming with tools):    {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! The fix appears to be working correctly.")
    else:
        print("\n⚠️  SOME TESTS FAILED! The fix may need additional work.")