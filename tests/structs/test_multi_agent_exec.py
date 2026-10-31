import asyncio

import pytest

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
