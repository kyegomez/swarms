import json
import types
from unittest.mock import patch

import pytest

from swarms import Agent, PlannerWorkerSwarm


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


def test_planner_that_answers_in_text_still_yields_its_tasks():
    swarm = PlannerWorkerSwarm(agents=[_usage_agent("Worker", 0, 0)])
    plan = json.dumps(
        {
            "plan": "p",
            "tasks": [{"title": "Write", "description": "d"}],
        }
    )
    prose = "Coverage catches regressions early. " * 30

    def planner_says(text):
        planner = _usage_agent(
            "Planner", 0, 0, [{"role": "Planner", "content": text}]
        )
        swarm._create_planner_agent = lambda name="Planner": planner

    for reply in (plan, f"```json\n{plan}\n```"):
        planner_says(reply)
        assert [t.title for t in swarm._run_planner("Plan.")] == [
            "Write"
        ]

    planner_says(prose)
    with pytest.raises(ValueError) as err:
        swarm._run_planner("Plan.")
    assert prose[:200] in str(err.value)
    assert len(str(err.value)) < len(prose)
