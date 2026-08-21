import pytest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure examples/tools is in path to import feedo_tools
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../examples/tools/feedo")))

try:
    from feedo_tools import FeedoMemoryTools
except ImportError:
    FeedoMemoryTools = None

@pytest.fixture
def mock_feedo_tools():
    if FeedoMemoryTools is None:
        pytest.skip("FeedoMemoryTools not available")
        
    with patch("feedo_tools.FeedoMemory") as mock_memory_cls:
        # Mock os.getenv to avoid "FEEDO_USAGE_KEY is required" error
        with patch.dict(os.environ, {"FEEDO_USAGE_KEY": "dummy_key"}):
            tools = FeedoMemoryTools()
            tools.memory = mock_memory_cls.return_value
            return tools

def test_add_memory(mock_feedo_tools):
    mock_feedo_tools.memory.add = MagicMock(return_value="mem_123")
    res = mock_feedo_tools.add_memory("Hello", topic="test")
    
    mock_feedo_tools.memory.add.assert_called_once_with("Hello", metadata={"topic": "test"})
    assert "mem_123" in res

def test_search_memory(mock_feedo_tools):
    mock_feedo_tools.memory.search = MagicMock(return_value=[{"text": "Hello World"}])
    res = mock_feedo_tools.search_memory("Hello")
    
    mock_feedo_tools.memory.search.assert_called_once_with("Hello", limit=5)
    assert "- Hello World" in res

def test_search_memory_empty(mock_feedo_tools):
    mock_feedo_tools.memory.search = MagicMock(return_value=[])
    res = mock_feedo_tools.search_memory("Nothing")
    assert "No relevant memories found" in res

def test_update_memory(mock_feedo_tools):
    mock_feedo_tools.memory.update = MagicMock(return_value="mem_456")
    res = mock_feedo_tools.update_memory("mem_123", "New text")
    
    mock_feedo_tools.memory.update.assert_called_once_with("mem_123", "New text")
    assert "mem_456" in res

def test_delete_memory(mock_feedo_tools):
    mock_feedo_tools.memory.delete = MagicMock()
    res = mock_feedo_tools.delete_memory("mem_123")
    
    mock_feedo_tools.memory.delete.assert_called_once_with("mem_123")
    assert "mem_123" in res
