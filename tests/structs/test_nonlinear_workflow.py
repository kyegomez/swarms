from unittest.mock import patch, MagicMock
from swarms.structs.nonlinear_workflow import NonLinearWorkflow, Task


class MockTask(Task):
    def can_execute(self):
        return True

    def execute(self):
        return "Task executed"


def test_nonlinearworkflow_initialization():
    agents = MagicMock()
    iters_per_task = MagicMock()
    workflow = NonLinearWorkflow(agents, iters_per_task)
    assert isinstance(workflow, NonLinearWorkflow)
    assert workflow.agents == agents
    assert workflow.tasks == []


def test_nonlinearworkflow_add():
    agents = MagicMock()
    iters_per_task = MagicMock()
    workflow = NonLinearWorkflow(agents, iters_per_task)
    task = MockTask("task1")
    workflow.add(task)
    assert workflow.tasks == [task]


@patch("your_module.NonLinearWorkflow.is_finished")
@patch("your_module.NonLinearWorkflow.output_tasks")
def test_nonlinearworkflow_run(mock_output_tasks, mock_is_finished):
    agents = MagicMock()
    iters_per_task = MagicMock()
    workflow = NonLinearWorkflow(agents, iters_per_task)
    task = MockTask("task1")
    workflow.add(task)
    mock_is_finished.return_value = False
    mock_output_tasks.return_value = [task]
    workflow.run()
    assert mock_output_tasks.called


def test_nonlinearworkflow_output_tasks():
    agents = MagicMock()
    iters_per_task = MagicMock()
    workflow = NonLinearWorkflow(agents, iters_per_task)
    task = MockTask("task1")
    workflow.add(task)
    assert workflow.output_tasks() == [task]


def test_nonlinearworkflow_to_graph():
    agents = MagicMock()
    iters_per_task = MagicMock()
    workflow = NonLinearWorkflow(agents, iters_per_task)
    task = MockTask("task1")
    workflow.add(task)
    assert workflow.to_graph() == {"task1": set()}


def test_nonlinearworkflow_order_tasks():
    agents = MagicMock()
    iters_per_task = MagicMock()
    workflow = NonLinearWorkflow(agents, iters_per_task)
    task = MockTask("task1")
    workflow.add(task)
    assert workflow.order_tasks() == [task]
