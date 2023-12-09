import pytest
from unittest.mock import Mock
from swarms.workers.worker_agent_ultra import WorkerUltraNode  # import your module here


def test_create_agent():
    mock_llm = Mock()
    mock_toolset = {"test_toolset": Mock()}
    mock_vectorstore = Mock()
    worker = WorkerUltraNode(mock_llm, mock_toolset, mock_vectorstore)
    worker.create_agent()
    assert worker.agent is not None


@pytest.mark.parametrize("invalid_toolset", [123, "string", 0.45])
def test_add_toolset_invalid(invalid_toolset):
    mock_llm = Mock()
    mock_toolset = {"test_toolset": Mock()}
    mock_vectorstore = Mock()
    worker = WorkerUltraNode(mock_llm, mock_toolset, mock_vectorstore)
    with pytest.raises(TypeError):
        worker.add_toolset(invalid_toolset)


@pytest.mark.parametrize("invalid_prompt", [123, None, "", []])
def test_run_invalid_prompt(invalid_prompt):
    mock_llm = Mock()
    mock_toolset = {"test_toolset": Mock()}
    mock_vectorstore = Mock()
    worker = WorkerUltraNode(mock_llm, mock_toolset, mock_vectorstore)
    with pytest.raises((TypeError, ValueError)):
        worker.run(invalid_prompt)


def test_run_valid_prompt(mocker):
    mock_llm = Mock()
    mock_toolset = {"test_toolset": Mock()}
    mock_vectorstore = Mock()
    worker = WorkerUltraNode(mock_llm, mock_toolset, mock_vectorstore)
    mocker.patch.object(worker, "create_agent")
    assert worker.run("Test prompt") == "Task completed by WorkerNode"


def test_worker_node():
    worker = worker_ultra_node("test-key")
    assert isinstance(worker, WorkerUltraNode)


def test_worker_node_no_key():
    with pytest.raises(ValueError):
        worker_ultra_node(None)
