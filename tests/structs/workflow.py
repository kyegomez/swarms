from unittest.mock import patch, MagicMock
from swarms.structs.workflow import Workflow


def test_workflow_initialization():
    agent = MagicMock()
    workflow = Workflow(agent)
    assert isinstance(workflow, Workflow)
    assert workflow.agent == agent
    assert workflow.tasks == []
    assert workflow.parallel is False


def test_workflow_add():
    agent = MagicMock()
    workflow = Workflow(agent)
    task = workflow.add("What's the weather in miami")
    assert isinstance(task, Workflow.Task)
    assert task.task == "What's the weather in miami"
    assert task.parents == []
    assert task.children == []
    assert task.output is None
    assert task.structure == workflow


def test_workflow_first_task():
    agent = MagicMock()
    workflow = Workflow(agent)
    assert workflow.first_task() is None
    workflow.add("What's the weather in miami")
    assert workflow.first_task().task == "What's the weather in miami"


def test_workflow_last_task():
    agent = MagicMock()
    workflow = Workflow(agent)
    assert workflow.last_task() is None
    workflow.add("What's the weather in miami")
    assert workflow.last_task().task == "What's the weather in miami"


@patch("your_module.Workflow.__run_from_task")
def test_workflow_run(mock_run_from_task):
    agent = MagicMock()
    workflow = Workflow(agent)
    workflow.add("What's the weather in miami")
    workflow.run()
    mock_run_from_task.assert_called_once()


def test_workflow_context():
    agent = MagicMock()
    workflow = Workflow(agent)
    task = workflow.add("What's the weather in miami")
    assert workflow.context(task) == {
        "parent_output": None,
        "parent": None,
        "child": None,
    }


@patch("your_module.Workflow.Task.execute")
def test_workflow___run_from_task(mock_execute):
    agent = MagicMock()
    workflow = Workflow(agent)
    task = workflow.add("What's the weather in miami")
    mock_execute.return_value = "Sunny"
    workflow.__run_from_task(task)
    mock_execute.assert_called_once()
