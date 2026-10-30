import json
import pytest
from swarms.structs.agent import Agent
from swarms.structs.multi_agent_router import MultiAgentRouter


def create_mock_agent(name):
    agent = Agent(agent_name=name, model_name="gpt-5.4", max_loops=1, print_on=False)
    agent.run = lambda task, *args, **kwargs: f"{name} solved: {task}"
    return agent


def test_sequential_runs_do_not_share_conversation():
    """Sequential router runs must not leak prior tasks into subsequent run conversations."""
    agent1 = create_mock_agent("Agent1")
    router = MultiAgentRouter(agents=[agent1], output_type="list")

    def fake_caller_run(task):
        return json.dumps({"handoffs": [{"agent_name": "Agent1", "task": task}]})

    router.function_caller.run = fake_caller_run

    res1 = router.run("Task 1")
    assert len(res1) == 2
    assert res1[0]["content"] == "Task 1"
    assert res1[1]["content"] == "Agent1 solved: Task 1"

    res2 = router.run("Task 2")
    assert len(res2) == 2
    contents = [m["content"] for m in res2]
    assert "Task 1" not in contents
    assert res2[0]["content"] == "Task 2"
    assert res2[1]["content"] == "Agent1 solved: Task 2"


def test_concurrent_batch_run_does_not_interleave_conversations():
    """Concurrent batch runs must not leak conversation history across threads."""
    agent1 = create_mock_agent("Agent1")
    router = MultiAgentRouter(agents=[agent1], output_type="list")

    def fake_caller_run(task):
        return json.dumps({"handoffs": [{"agent_name": "Agent1", "task": task}]})

    router.function_caller.run = fake_caller_run

    tasks = ["Task Alpha", "Task Beta", "Task Gamma"]
    results = router.concurrent_batch_run(tasks)
    assert len(results) == 3

    for task_name, res in zip(tasks, results):
        assert len(res) == 2
        assert res[0]["content"] == task_name
        assert res[1]["content"] == f"Agent1 solved: {task_name}"
