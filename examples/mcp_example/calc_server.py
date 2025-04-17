
from fastmcp import FastMCP
from typing import Dict, Any

# Initialize MCP server for business calculations
mcp = FastMCP("Calc-Server")

@mcp.tool()
def profit_margin(revenue: float, cost: float) -> Dict[str, Any]:
    """Calculate profit margin from revenue and cost"""
    try:
        profit = revenue - cost
        margin = (profit / revenue) * 100
        return {
            "profit": profit,
            "margin_percentage": margin,
            "summary": f"On revenue of ${revenue:.2f} and costs of ${cost:.2f}, profit is ${profit:.2f} with a margin of {margin:.1f}%"
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def break_even_point(fixed_costs: float, price_per_unit: float, cost_per_unit: float) -> Dict[str, Any]:
    """Calculate break-even point"""
    try:
        bep = fixed_costs / (price_per_unit - cost_per_unit)
        return {
            "break_even_units": bep,
            "summary": f"You need to sell {bep:.0f} units to break even"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("Starting Business Calculator Server on port 6275...")
    mcp.run(transport="sse", transport_kwargs={"host": "0.0.0.0", "port": 6275})
