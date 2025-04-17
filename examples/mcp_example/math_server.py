
from fastmcp import FastMCP
from typing import Dict, Any, Optional

# Initialize MCP server
mcp = FastMCP("Math-Server")

# Add tool documentation and type hints
@mcp.tool()
def add(a: int, b: int) -> Dict[str, Any]:
    """Add two numbers together
    
    Args:
        a (int): First number to add
        b (int): Second number to add
        
    Returns:
        Dict[str, Any]: Result dictionary containing sum and metadata
    """
    try:
        result = a + b
        return {
            "status": "success",
            "result": result,
            "message": f"Successfully added {a} and {b}"
        }
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e),
            "message": "Failed to perform addition"
        }

@mcp.tool()
def multiply(a: int, b: int) -> Dict[str, Any]:
    """Multiply two numbers together
    
    Args:
        a (int): First number to multiply 
        b (int): Second number to multiply
        
    Returns:
        Dict[str, Any]: Result dictionary containing product and metadata
    """
    try:
        result = a * b
        return {
            "status": "success",
            "result": result,
            "message": f"Successfully multiplied {a} and {b}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to perform multiplication"
        }

@mcp.tool()
def get_available_operations() -> Dict[str, Any]:
    """Get list of available mathematical operations
    
    Returns:
        Dict[str, Any]: Dictionary containing available operations and their descriptions
    """
    return {
        "status": "success",
        "operations": {
            "add": "Add two numbers together",
            "multiply": "Multiply two numbers together"
        }
    }

if __name__ == "__main__":
    print("Starting Math Server...")
    print("Available operations:", get_available_operations())
    mcp.run(host="0.0.0.0", port=6274, transport="sse")
