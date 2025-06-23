from mcp.server.fastmcp import FastMCP

mcp = FastMCP("NewsServer")

mcp.settings.port = 9001

@mcp.tool(name="get_news", description="Return simple news headline")
def get_news(topic: str) -> str:
    return f"Latest {topic} news headline"

if __name__ == "__main__":
    mcp.run(transport="sse")
