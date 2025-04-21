import os
import sys
from loguru import logger
from swarms import Agent
from swarms.prompts.agent_prompts import MATH_AGENT_PROMPT
from swarms.tools.mcp_integration import MCPServerSseParams

# Configure API key

# Configure logging
logger.remove()
logger.add(sys.stdout, level="DEBUG", format="{time} | {level} | {message}")

# Define a simpler prompt that focuses on math operations
SIMPLE_MATH_PROMPT = """
You are a math calculator assistant that uses external tools.
When asked for calculations, extract the numbers and use the appropriate tool.
Available tools:
- add: For addition
- multiply: For multiplication
- divide: For division
Keep your responses concise and focused on the calculation.
"""

def main():
    print("=== MINIMAL MCP AGENT INTEGRATION TEST ===")
    
    # Properly configured MCP parameters
    mcp_params = MCPServerSseParams(
        url="http://127.0.0.1:8000",
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        },
        timeout=30.0,  # Increased timeout
        sse_read_timeout=60.0
    )
    
    agent = Agent(
        agent_name="MCP Test Agent",
        system_prompt=SIMPLE_MATH_PROMPT,  # Using simpler prompt
        mcp_servers=[mcp_params],
        model_name="gpt-4o-mini",
        max_loops=2,  # Allow for retry
        verbose=True
    )
    
    print("\nAgent created successfully! Type 'exit' to quit.")
    while True:
        query = input("\nMath query: ").strip()
        if query.lower() == "exit":
            break
        
        print(f"\nProcessing: {query}")
        try:
            result = agent.run(query)
            print(f"\nResult: {result}")
        except Exception as e:
            print(f"\nError processing query: {str(e)}")

if __name__ == "__main__":
    main()