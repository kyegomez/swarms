import types
from unittest.mock import MagicMock, patch

import pytest

from swarms import Agent, PlannerWorkerSwarm
from swarms.structs.planner_worker_swarm import WorkerPool


def _usage_agent(name, input_tokens, output_tokens, output="ok"):
    """An offline agent whose run reports a fixed usage through the real hook."""
    agent = Agent(agent_name=name, max_loops=1, print_on=False)

    def run(self, task=None, *args, **kwargs):
        self._add_usage(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
        )
        return output

    agent.run = types.MethodType(run, agent)
    return agent


def test_usage_counts_the_single_use_planner_and_judge():
    """The planner and judge are built per call, and their usage still counts."""
    swarm = PlannerWorkerSwarm(
        agents=[_usage_agent("Worker", 1, 1)], max_loops=1
    )
    single_use = iter(
        [
            _usage_agent(
                "Planner", 100, 10, {"plan": "p", "tasks": []}
            ),
            _usage_agent("CycleJudge", 50, 5, "not a verdict"),
        ]
    )

    with patch(
        "swarms.structs.planner_worker_swarm.Agent",
        side_effect=lambda *a, **kw: next(single_use),
    ):
        swarm.run("Plan a product launch.")

    assert swarm.usage == {
        "input_tokens": 150,
        "output_tokens": 15,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 165,
    }


def test_planner_request_forces_the_plan_tool(monkeypatch):
    sent = []

    def capture(**params):
        sent.append(params)
        raise RuntimeError("captured")

    monkeypatch.setattr(
        "swarms.utils.litellm_wrapper.completion", capture
    )
    swarm = PlannerWorkerSwarm(agents=[_usage_agent("Worker", 1, 1)])

    with pytest.raises(RuntimeError):
        swarm._create_planner_agent().llm.run(task="plan")

    assert sent[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "PlannerTaskSpec"},
    }


def test_judge_request_forces_the_verdict_tool():
    swarm = PlannerWorkerSwarm(agents=[_usage_agent("Worker", 1, 1)])

    with patch(
        "swarms.structs.planner_worker_swarm.Agent",
        side_effect=RuntimeError,
    ) as built, pytest.raises(RuntimeError):
        swarm._run_judge()

    assert built.call_args.kwargs["llm_args"]["tool_choice"] == {
        "type": "function",
        "function": {"name": "CycleVerdict"},
    }


def test_judge_gets_the_goal_and_each_result_once():
    plan = {
        "plan": "p",
        "tasks": [{"title": "t1", "description": "d"}],
    }
    judge = MagicMock(usage={})
    judge.run.return_value = "not a verdict"
    worker = _usage_agent("Worker", 1, 1, "WORKER-RESULT")
    swarm = PlannerWorkerSwarm(agents=[worker])

    with patch(
        "swarms.structs.planner_worker_swarm.Agent",
        side_effect=[_usage_agent("Planner", 1, 1, plan), judge],
    ):
        swarm.run("TASK-MARK")

    text = str(judge.run.call_args.kwargs)
    counts = (text.count("TASK-MARK"), text.count("WORKER-RESULT"))
    assert counts == (1, 1)


def test_worker_pool_defaults_to_one_worker_per_agent():
    """Workers wait on model calls, so the CPU count does not cap the pool."""
    agents = [object()] * 8

    with patch("os.cpu_count", return_value=2):
        assert WorkerPool(agents, None, None).max_workers == 8
        assert (
            WorkerPool(agents, None, None, max_workers=3).max_workers
            == 3
        )


def test_worker_gets_the_original_task_with_its_step():
    plan = {
        "plan": "p",
        "tasks": [{"title": "Compare pricing", "description": "d"}],
    }
    judge = MagicMock(usage={})
    judge.run.return_value = "not a verdict"
    worker = MagicMock(agent_name="Worker", usage={})
    worker.run.return_value = "ok"
    swarm = PlannerWorkerSwarm(agents=[worker])

    with patch(
        "swarms.structs.planner_worker_swarm.Agent",
        side_effect=[_usage_agent("Planner", 1, 1, plan), judge],
    ):
        swarm.run("TASK-MARK")

    prompt = worker.run.call_args.kwargs["task"]
    assert "TASK-MARK" in prompt
    assert "Compare pricing" in prompt
