import pytest
from swarms.structs import NonlinearWorkflow, Task
from swarms.models import OpenAIChat


class TestNonlinearWorkflow:
    def test_add_task(self):
        llm = OpenAIChat(openai_api_key="")
        task = Task(llm, "What's the weather in miami")
        workflow = NonlinearWorkflow()
        workflow.add(task)
        assert task.name in workflow.tasks
        assert task.name in workflow.edges

    def test_run_without_tasks(self):
        workflow = NonlinearWorkflow()
        # No exception should be raised
        workflow.run()

    def test_run_with_single_task(self):
        llm = OpenAIChat(openai_api_key="")
        task = Task(llm, "What's the weather in miami")
        workflow = NonlinearWorkflow()
        workflow.add(task)
        # No exception should be raised
        workflow.run()

    def test_run_with_circular_dependency(self):
        llm = OpenAIChat(openai_api_key="")
        task1 = Task(llm, "What's the weather in miami")
        task2 = Task(llm, "What's the weather in new york")
        workflow = NonlinearWorkflow()
        workflow.add(task1, task2.name)
        workflow.add(task2, task1.name)
        with pytest.raises(
            Exception, match="Circular dependency detected"
        ):
            workflow.run()

    def test_run_with_stopping_token(self):
        llm = OpenAIChat(openai_api_key="")
        task1 = Task(llm, "What's the weather in miami")
        task2 = Task(llm, "What's the weather in new york")
        workflow = NonlinearWorkflow(stopping_token="stop")
        workflow.add(task1)
        workflow.add(task2)
        # Assuming that task1's execute method returns "stop"
        # No exception should be raised
        workflow.run()
