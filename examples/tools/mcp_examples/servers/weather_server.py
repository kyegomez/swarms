from mcp.server.fastmcp import FastMCP

mcp = FastMCP("WeatherServer")

mcp.settings.port = 8000

@mcp.tool(name="get_weather", description="Return simple weather info")
def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny 22°C"

if __name__ == "__main__":
    mcp.run(transport="sse")
