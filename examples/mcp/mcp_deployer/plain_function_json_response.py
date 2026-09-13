import socket

from swarms import MCPDeployer
from swarms.tools.mcp_manager import MCPManager


def analyse(task: str) -> dict:
    """Trivial text statistics for the task string."""
    words = task.split()
    return {
        "characters": len(task),
        "words": len(words),
        "longest_word": max(words, key=len) if words else "",
    }


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    deployer = MCPDeployer(
        analyse,
        api_keys=["stats-key"],
        port=free_port(),
        json_response=True,  # plain JSON replies instead of an event stream
        timeout=5,
    )
    with deployer:
        client = MCPManager(mcp_url=deployer.url, api_key="stats-key")
        print("tools:", client.list_tool_names())
        result = client.call_tool(
            "analyse",
            {"task": "the quick brown fox jumps over the lazy dog"},
        )
        print(result["result"])
