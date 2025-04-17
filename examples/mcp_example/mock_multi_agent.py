
import asyncio
from swarms import Agent
from swarms.tools.mcp_integration import MCPServerSseParams
import logging

class MathAgent:
    def __init__(self, name: str, server_url: str):
        self.server = MCPServerSseParams(
            url=server_url,
            headers={"Content-Type": "application/json"}
        )
        
        self.agent = Agent(
            agent_name=name,
            agent_description="Math processing agent",
            system_prompt=f"You are {name}, a math processing agent. You have access to these mathematical operations ONLY: addition, multiplication, and division. Only suggest calculations using these available tools. Do not attempt to solve problems requiring other operations like percentages, square roots, or advanced math. When users ask about capabilities, list only the basic operations you can perform.",
            max_loops=1,
            mcp_servers=[self.server],
            streaming_on=False,
            model_name="gpt-4o-mini",
            temperature=0.1,
            max_tokens=1000
        )

    async def process(self, task: str):
        try:
            response = await self.agent.arun(task)
            return {
                "agent": self.agent.agent_name,
                "task": task,
                "response": str(response)
            }
        except Exception as e:
            logging.error(f"Error in {self.agent.agent_name}: {str(e)}")
            return {
                "agent": self.agent.agent_name,
                "task": task,
                "error": str(e)
            }

class MultiAgentMathSystem:
    def __init__(self):
        math_url = "http://0.0.0.0:8000"
        stock_url = "http://0.0.0.0:8001"
        self.calculator = MathAgent("Calculator", math_url)
        self.stock_analyst = MathAgent(
            "StockAnalyst", 
            stock_url,
            "Stock market analysis agent specializing in financial calculations and market analysis"
        )

    async def process_task(self, task: str):
        # Process with both agents
        results = await asyncio.gather(
            self.calculator.process(task),
            self.stock_analyst.process(task)
        )
        return results

    def run_interactive(self):
        print("\nMulti-Agent Math System")
        print("Enter 'exit' to quit")
        
        while True:
            try:
                user_input = input("\nEnter a math problem: ")
                if user_input.lower() == 'exit':
                    break

                results = asyncio.run(self.process_task(user_input))
                
                print("\nResult:")
                result = results[0]  # We now only have one result
                if "error" in result:
                    print(f"\nCalculator encountered an error: {result['error']}")
                else:
                    print(f"\nCalculation: {result['response']}")

            except Exception as e:
                print(f"System error: {str(e)}")

if __name__ == "__main__":
    system = MultiAgentMathSystem()
    system.run_interactive()
