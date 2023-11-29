import pytest
from unittest.mock import Mock
from swarms.swarms.orchestrate import Orchestrator


@pytest.fixture
def mock_agent():
    return Mock()


@pytest.fixture
def mock_task():
    return {"task_id": 1, "task_data": "data"}


@pytest.fixture
def mock_vector_db():
    return Mock()


@pytest.fixture
def orchestrator(mock_agent, mock_vector_db):
    agent_list = [mock_agent for _ in range(5)]
    task_queue = []
    return Orchestrator(
        mock_agent, agent_list, task_queue, mock_vector_db
    )


def test_assign_task(
    orchestrator, mock_agent, mock_task, mock_vector_db
):
    orchestrator.task_queue.append(mock_task)
    orchestrator.assign_task(0, mock_task)

    mock_agent.process_task.assert_called_once()
    mock_vector_db.add_documents.assert_called_once()


def test_retrieve_results(orchestrator, mock_vector_db):
    mock_vector_db.query.return_value = "expected_result"
    assert orchestrator.retrieve_results(0) == "expected_result"


def test_update_vector_db(orchestrator, mock_vector_db):
    data = {"vector": [0.1, 0.2, 0.3], "task_id": 1}
    orchestrator.update_vector_db(data)
    mock_vector_db.add_documents.assert_called_once_with(
        [data["vector"]], [str(data["task_id"])]
    )


def test_get_vector_db(orchestrator, mock_vector_db):
    assert orchestrator.get_vector_db() == mock_vector_db


def test_append_to_db(orchestrator, mock_vector_db):
    collection = "test_collection"
    result = "test_result"
    orchestrator.append_to_db(collection, result)
    mock_vector_db.append_document.assert_called_once_with(
        collection, result, id=str(id(result))
    )


def test_run(orchestrator, mock_agent, mock_vector_db):
    objective = "test_objective"
    collection = "test_collection"
    orchestrator.run(objective, collection)

    mock_agent.process_task.assert_called()
    mock_vector_db.append_document.assert_called()
