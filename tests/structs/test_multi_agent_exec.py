import asyncio
import threading
from types import SimpleNamespace

import pytest

from swarms.structs.agent import Agent
from swarms.structs.multi_agent_exec import (
    run_agents_concurrently_async,
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


@pytest.mark.parametrize(
    "thread_backed_arun, max_concurrency, peak",
    [(True, None, 40), (False, None, 40), (True, 4, 4)],
)
def test_run_agents_concurrently_async_runs_blocking_agents_up_to_the_bound(
    thread_backed_arun, max_concurrency, peak
):
    barrier = threading.Barrier(peak, timeout=5)
    lock = threading.Lock()
    gauge = {"active": 0, "peak": 0}

    def run(task=None, img=None, *args, **kwargs):
        with lock:
            gauge["active"] += 1
            gauge["peak"] = max(gauge["peak"], gauge["active"])
        barrier.wait()
        with lock:
            gauge["active"] -= 1
        return task

    agents = []
    for _ in range(40):
        agent = (
            Agent.__new__(Agent)
            if thread_backed_arun
            else SimpleNamespace()
        )
        agent.run = run
        agents.append(agent)

    results = asyncio.run(
        run_agents_concurrently_async(
            agents, "t", max_concurrency=max_concurrency
        )
    )

    assert results == ["t"] * 40
    assert gauge["peak"] == peak
