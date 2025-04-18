
from swarms import Agent
from swarms.tools.mcp_integration import MCPServerSseParams
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPClient:
    def __init__(self):
        # Configure MCP servers
        self.math_server = MCPServerSseParams(
            url="http://0.0.0.0:8000/mcp",
            headers={"Content-Type": "application/json"},
            timeout=5.0,
            sse_read_timeout=30.0
        )
        
        self.stock_server = MCPServerSseParams(
            url="http://0.0.0.0:8001/mcp",
            headers={"Content-Type": "application/json"},
            timeout=5.0,
            sse_read_timeout=30.0
        )

        # Initialize agents with specific servers
        self.math_agent = Agent(
            agent_name="Math Agent",
            system_prompt="""You are a mathematical computation specialist with access to the following capabilities through the MCP server:
            - Addition of two numbers
            - Multiplication of two numbers  
            - Power/exponent calculations
            - Square root calculations
            
            Always follow these rules:
            1. Only use tools that are available from the MCP server
            2. First list the available tools when asked
            3. Explain your mathematical approach before using tools
            4. Provide clear step-by-step explanations of calculations""",
            mcp_servers=[self.math_server],
            max_loops=1,
            streaming_on=True,
            model_name="gpt-4o-mini",
            temperature=0.1
        )

        self.stock_agent = Agent(
            agent_name="Stock Agent",
            system_prompt="""You are a stock market analysis specialist with access to the following capabilities through the MCP server:
            - Get current stock prices
            - Get trading volumes
            - Calculate market capitalization
            - Generate price statistics across multiple stocks
            
            Always follow these rules:
            1. Only use tools that are available from the MCP server
            2. First list the available tools when asked
            3. Explain your analysis approach before using tools
            4. Provide clear explanations of market metrics""",
            mcp_servers=[self.stock_server],
            max_loops=1,
            streaming_on=True,
            model_name="gpt-4o-mini",
            temperature=0.1
        )

    async async def process_query(self, query: str):
        try:
            if query.lower() in ["capabilities", "what can you do", "what kind of problems you can solve"]:
                return [
                    {"agent": "Math Agent", "response": """I can help with mathematical computations including:
- Addition of two numbers
- Multiplication of two numbers
- Power/exponent calculations  
- Square root calculations"""},
                    {"agent": "Stock Agent", "response": """I can help with stock market analysis including:
- Get current stock prices
- Get trading volumes
- Calculate market capitalization
- Generate price statistics across stocks"""}
                ]
            
            # Run agents concurrently
            results = await asyncio.gather(
                self.math_agent.arun(query),
                self.stock_agent.arun(query)
            )
            
            # Format results
            formatted_results = []
            for idx, result in enumerate(results):
                agent_name = "Math Agent" if idx == 0 else "Stock Agent"
                if isinstance(result, dict):
                    formatted_results.append({
                        "agent": agent_name,
                        "response": result.get("response", str(result))
                    })
                else:
                    formatted_results.append({
                        "agent": agent_name,
                        "response": str(result)
                    })
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return None

def main():
    client = MCPClient()
    
    print("\nAvailable Agents and Capabilities:")
    print("\nMath Agent:")
    print("- Addition of two numbers")
    print("- Multiplication of two numbers")
    print("- Power/exponent calculations")
    print("- Square root calculations")
    
    print("\nStock Agent:")
    print("- Get current stock prices")
    print("- Get trading volumes") 
    print("- Calculate market capitalization")
    print("- Generate price statistics")
    
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
            
        results = asyncio.run(client.process_query(query))
        
        if results:
            print("\nResults:")
            for result in results:
                if not result:
                    continue
                print(f"\n{result['agent']}:")
                print("-" * 50)
                print(result['response'])
                print("-" * 50)

if __name__ == "__main__":
    main()
