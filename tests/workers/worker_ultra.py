import pytest
from unittest.mock import Mock, patch
from swarms.workers.worker_agent_ultra import (
    WorkerUltraNode,
    WorkerUltraNodeInitializer,
)


@pytest.fixture
def llm_mock():
    return Mock()


@pytest.fixture
def toolsets_mock():
    return Mock()


@pytest.fixture
def vectorstore_mock():
    return Mock()


@pytest.fixture
def worker_ultra_node(llm_mock, toolsets_mock, vectorstore_mock):
    return WorkerUltraNode(llm_mock, toolsets_mock, vectorstore_mock)


def test_worker_ultra_node_create_agent(worker_ultra_node):
    with patch("yourmodule.AutoGPT.from_llm_and_tools") as mock_method:
        worker_ultra_node.create_agent()
        mock_method.assert_called_once()


def test_worker_ultra_node_add_toolset(worker_ultra_node):
    with pytest.raises(TypeError):
        worker_ultra_node.add_toolset("wrong_toolset")


def test_worker_ultra_node_run(worker_ultra_node):
    with patch.object(worker_ultra_node, "agent") as mock_agent:
        mock_agent.run.return_value = None
        result = worker_ultra_node.run("some prompt")
        assert result == "Task completed by WorkerNode"
        mock_agent.run.assert_called_once()


def test_worker_ultra_node_run_no_prompt(worker_ultra_node):
    with pytest.raises(ValueError):
        worker_ultra_node.run("")


@pytest.fixture
def worker_ultra_node_initializer():
    return WorkerUltraNodeInitializer("openai_api_key")


def test_worker_ultra_node_initializer_initialize_llm(worker_ultra_node_initializer):
    with patch("yourmodule.ChatOpenAI") as mock_llm:
        worker_ultra_node_initializer.initialize_llm(mock_llm)
        mock_llm.assert_called_once()


def test_worker_ultra_node_initializer_initialize_toolsets(
    worker_ultra_node_initializer,
):
    with patch("yourmodule.Terminal"), patch("yourmodule.CodeEditor"), patch(
        "yourmodule.RequestsGet"
    ), patch("yourmodule.ExitConversation"):
        toolsets = worker_ultra_node_initializer.initialize_toolsets()
        assert len(toolsets) == 4


def test_worker_ultra_node_initializer_initialize_vectorstore(
    worker_ultra_node_initializer,
):
    with patch("yourmodule.OpenAIEmbeddings"), patch(
        "yourmodule.fauss.IndexFlatL2"
    ), patch("yourmodule.FAISS"), patch("yourmodule.InMemoryDocstore"):
        vectorstore = worker_ultra_node_initializer.initialize_vectorstore()
        assert vectorstore is not None


def test_worker_ultra_node_initializer_create_worker_node(
    worker_ultra_node_initializer,
):
    with patch.object(worker_ultra_node_initializer, "initialize_llm"), patch.object(
        worker_ultra_node_initializer, "initialize_toolsets"
    ), patch.object(worker_ultra_node_initializer, "initialize_vectorstore"):
        worker_node = worker_ultra_node_initializer.create_worker_node()
        assert worker_node is not None
