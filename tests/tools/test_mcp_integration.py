
import pytest
from swarms.tools.mcp_integration import MCPServerSseParams, mcp_flow

def test_mcp_flow():
    params = MCPServerSseParams(
        url="http://localhost:6274",
        headers={"Content-Type": "application/json"}
    )
    
    function_call = {
        "tool_name": "test_tool",
        "args": {"param1": "value1"}
    }
    
    try:
        result = mcp_flow(params, function_call)
        assert isinstance(result, str)
    except Exception as e:
        pytest.fail(f"MCP flow failed: {e}")

def test_mcp_invalid_params():
    with pytest.raises(Exception):
        mcp_flow(None, {})
