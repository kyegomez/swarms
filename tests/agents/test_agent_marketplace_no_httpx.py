import ast
from pathlib import Path


def test_no_module_level_httpx_in_marketplace_handler():
    """Verify swarms/agents/agent_marketplace_handler.py does not import httpx at module level."""
    handler_path = Path(__file__).resolve().parents[2] / "swarms" / "agents" / "agent_marketplace_handler.py"
    tree = ast.parse(handler_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "httpx", "Top-level 'import httpx' found in agent_marketplace_handler.py"
        elif isinstance(node, ast.ImportFrom):
            assert node.module != "httpx", f"Top-level 'from httpx import ...' found in agent_marketplace_handler.py: {node.module}"


def test_agent_py_no_module_level_marketplace_handler():
    """Verify swarms/structs/agent.py does not import AgentMarketplaceHandler at module level."""
    agent_path = Path(__file__).resolve().parents[2] / "swarms" / "structs" / "agent.py"
    tree = ast.parse(agent_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            assert "agent_marketplace_handler" not in (node.module or ""), (
                "Top-level 'from ...agent_marketplace_handler import ...' found in agent.py"
            )
