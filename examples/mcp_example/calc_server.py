
import asyncio
from mcp import run
from swarms.utils.litellm_wrapper import LiteLLM

def calculate_compound_interest(principal: float, rate: float, time: float) -> float:
    """Calculate compound interest."""
    return principal * (1 + rate/100) ** time - principal

def calculate_simple_interest(principal: float, rate: float, time: float) -> float:
    """Calculate simple interest."""
    return (principal * rate * time) / 100

# Create tool registry  
tools = {
    "calculate_compound_interest": calculate_compound_interest,
    "calculate_simple_interest": calculate_simple_interest,
}

async def handle_tool(name: str, args: dict) -> dict:
    """Handle tool execution."""
    try:
        result = tools[name](**args)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("Starting Calculation Server on port 6275...")
    llm = LiteLLM()
    run(transport="sse", port=6275, tool_handler=handle_tool)
