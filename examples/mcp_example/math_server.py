import asyncio
from mcp import run
from swarms.utils.litellm_wrapper import LiteLLM

def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# Create tool registry
tools = {
    "add": add,
    "subtract": subtract, 
    "multiply": multiply,
    "divide": divide
}

async def handle_tool(name: str, args: dict) -> dict:
    """Handle tool execution."""
    try:
        result = tools[name](**args)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("Starting Math Server on port 6274...")
    llm = LiteLLM()
    run(transport="sse", port=6274, tool_handler=handle_tool)