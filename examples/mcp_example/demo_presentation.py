
"""
MCP Integration Demo Script
This script demonstrates the full MCP integration workflow
"""
import asyncio
import time
from swarms.tools.mcp_integration import MCPServerSseParams
from examples.mcp_example.mock_multi_agent import MultiAgentMathSystem

def print_section(title):
    print("\n" + "="*50)
    print(title)
    print("="*50 + "\n")

async def run_demo():
    print_section("1. Initializing Multi-Agent MCP System")
    system = MultiAgentMathSystem()
    
    print_section("2. Testing Basic Operations")
    results = await system.process_task("What operations can you perform?")
    for result in results:
        print(f"\n[{result['agent']}]")
        print(f"Response: {result['response']}")
    
    print_section("3. Testing Mathematical Operations")
    test_operations = [
        "5 plus 3",
        "10 times 4",
        "20 divide by 5"
    ]
    
    for operation in test_operations:
        print(f"\nTesting: {operation}")
        results = await system.process_task(operation)
        for result in results:
            if "error" not in result:
                print(f"[{result['agent']}]: {result['response']}")
    
    print_section("4. Testing Error Handling")
    results = await system.process_task("calculate square root of 16")
    for result in results:
        print(f"\n[{result['agent']}]")
        if "error" in result:
            print(f"Error handled: {result['error']}")
        else:
            print(f"Response: {result['response']}")

if __name__ == "__main__":
    print("\nMCP Integration Demonstration")
    print("Running comprehensive demo of MCP functionality\n")
    asyncio.run(run_demo())
