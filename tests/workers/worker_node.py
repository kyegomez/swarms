import pytest
from unittest.mock import MagicMock, patch
from swarms.worker.worker_node import (
    WorkerNodeInitializer,
    WorkerNode,
)  # replace your_module with actual module name


# Mock Tool for testing
class MockTool(Tool):
    pass


# Fixture for llm
@pytest.fixture
def mock_llm():
    return MagicMock()


# Fixture for vectorstore
@pytest.fixture
def mock_vectorstore():
    return MagicMock()


# Fixture for Tools
@pytest.fixture
def mock_tools():
    return [MockTool(), MockTool(), MockTool()]


# Fixture for WorkerNodeInitializer
@pytest.fixture
def worker_node(mock_llm, mock_tools, mock_vectorstore):
    return WorkerNodeInitializer(
        llm=mock_llm, tools=mock_tools, vectorstore=mock_vectorstore
    )


# Fixture for WorkerNode
@pytest.fixture
def mock_worker_node():
    return WorkerNode(openai_api_key="test_api_key")


# WorkerNodeInitializer Tests
def test_worker_node_init(worker_node):
    assert worker_node.llm is not None
    assert worker_node.tools is not None
    assert worker_node.vectorstore is not None


def test_worker_node_create_agent(worker_node):
    with patch.object(AutoGPT, "from_llm_and_tools") as mock_method:
        worker_node.create_agent()
        mock_method.assert_called_once()


def test_worker_node_add_tool(worker_node):
    initial_tools_count = len(worker_node.tools)
    new_tool = MockTool()
    worker_node.add_tool(new_tool)
    assert len(worker_node.tools) == initial_tools_count + 1


def test_worker_node_run(worker_node):
    with patch.object(worker_node.agent, "run") as mock_run:
        worker_node.run(prompt="test prompt")
        mock_run.assert_called_once()


# WorkerNode Tests
def test_worker_node_llm(mock_worker_node):
    with patch.object(mock_worker_node, "initialize_llm") as mock_method:
        mock_worker_node.initialize_llm(llm_class=MagicMock(), temperature=0.5)
        mock_method.assert_called_once()


def test_worker_node_tools(mock_worker_node):
    with patch.object(mock_worker_node, "initialize_tools") as mock_method:
        mock_worker_node.initialize_tools(llm_class=MagicMock())
        mock_method.assert_called_once()


def test_worker_node_vectorstore(mock_worker_node):
    with patch.object(mock_worker_node, "initialize_vectorstore") as mock_method:
        mock_worker_node.initialize_vectorstore()
        mock_method.assert_called_once()


def test_worker_node_create_worker_node(mock_worker_node):
    with patch.object(mock_worker_node, "create_worker_node") as mock_method:
        mock_worker_node.create_worker_node()
        mock_method.assert_called_once()
