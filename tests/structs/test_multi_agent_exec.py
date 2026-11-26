import asyncio
import os
import threading
from types import SimpleNamespace

import pytest

from swarms.structs.multi_agent_exec import (
    batched_grid_agent_execution,
    get_agents_info,
    run_agents_concurrently,
    run_agents_concurrently_async,
    run_agents_with_different_tasks,
    run_single_agent,
)


class AsyncAgent:
    def __init__(self, name, gauge):
        self.agent_name = name
        self.gauge = gauge

    def run(self, task=None, img=None, *args, **kwargs):
        raise AssertionError("ran on a thread instead of arun")

    async def arun(self, task=None, img=None, *args, **kwargs):
        self.gauge["active"] += 1
        self.gauge["peak"] = max(
            self.gauge["peak"], self.gauge["active"]
        )
        await asyncio.sleep(0)
        self.gauge["active"] -= 1
        return f"{self.agent_name}:{task}"


@pytest.mark.parametrize("max_concurrency, peak", [(None, 5), (2, 2)])
def test_run_agents_concurrently_async_awaits_arun_within_the_bound(
    max_concurrency, peak
):
    gauge = {"active": 0, "peak": 0}
    agents = [AsyncAgent(f"a{i}", gauge) for i in range(5)]

    results = asyncio.run(
        run_agents_concurrently_async(
            agents, "t", max_concurrency=max_concurrency
        )
    )

    assert results == [f"a{i}:t" for i in range(5)]
    assert gauge["peak"] == peak


def test_default_pool_has_one_thread_per_agent():
    count = (os.cpu_count() or 1) + 5
    barrier = threading.Barrier(count, timeout=5)
    agent = SimpleNamespace(run=lambda task=None: barrier.wait())
    agents, tasks = [agent] * count, ["t"] * count
    done = set(range(count))

    assert set(run_agents_concurrently(agents, "t")) == done
    assert set(batched_grid_agent_execution(agents, tasks)) == done
    pairs = list(zip(agents, tasks))
    assert set(run_agents_with_different_tasks(pairs, count)) == done


def test_run_single_agent_forwards_positional_args():
    agent = SimpleNamespace(
        run=lambda task=None, img=None, **kwargs: (task, img)
    )
    assert run_single_agent(agent, "task", "img.png") == (
        "task",
        "img.png",
    )


def test_get_agents_info_layout():
    agents = [
        SimpleNamespace(
            agent_name=f"a{i}",
            agent_description="d",
            role="worker",
            model_name="m",
            max_loops=1,
        )
        for i in (1, 2)
    ]
    body = []
    for i in (1, 2):
        body += ["", f"[Agent {i}]", f"Name: a{i}", "Description: d"]
        body += ["Role: worker", "Model: m", "Max Loops: 1"]
    assert get_agents_info(agents) == "\n".join(
        ["Available Agents:", *body]
    )
    assert get_agents_info(agents, team_name="T") == "\n".join(
        ["Available Agents for Team: T", *body]
    )
