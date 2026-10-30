import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest

from swarms.structs.multi_agent_exec import (
    run_agent_async,
    run_agents_concurrently_async,
)


class MockNativeAsyncAgent:
    def __init__(self, name="async-agent"):
        self.agent_name = name
        self.arun = AsyncMock(return_value=f"async-ran:{name}")
        self.run = MagicMock(side_effect=RuntimeError("sync run called instead of arun"))


class MockSyncOnlyAgent:
    def __init__(self, name="sync-agent"):
        self.agent_name = name
        self.run = MagicMock(return_value=f"sync-ran:{name}")


@pytest.mark.asyncio
async def test_run_agent_async_prefers_native_arun():
    agent = MockNativeAsyncAgent("agent-1")
    res = await run_agent_async(agent, "test task")
    assert res == "async-ran:agent-1"
    agent.arun.assert_awaited_once_with("test task")
    agent.run.assert_not_called()


@pytest.mark.asyncio
async def test_run_agents_concurrently_async_semaphore_bounds_concurrency():
    active = 0
    max_active = 0
    lock = asyncio.Lock()

    class BoundedAgent:
        def __init__(self, idx):
            self.agent_name = f"agent-{idx}"

        async def arun(self, task, *args, **kwargs):
            nonlocal active, max_active
            async with lock:
                active += 1
                if active > max_active:
                    max_active = active
            await asyncio.sleep(0.02)
            async with lock:
                active -= 1
            return f"done-{self.agent_name}"

    agents = [BoundedAgent(i) for i in range(8)]
    results = await run_agents_concurrently_async(
        agents, "test task", max_concurrency=2
    )

    assert len(results) == 8
    assert max_active <= 2, f"Expected concurrency bounded to <= 2, got {max_active}"


@pytest.mark.asyncio
async def test_run_agent_async_fallback_to_sync_run():
    agent = MockSyncOnlyAgent("sync-1")
    res = await run_agent_async(agent, "sync task")
    assert res == "sync-ran:sync-1"
    agent.run.assert_called_once()


@pytest.mark.asyncio
async def test_run_agents_concurrently_async_forwards_args_kwargs():
    agent = MockNativeAsyncAgent("agent-fwd")
    await run_agents_concurrently_async(
        [agent], "task-x", "extra_arg", custom_kw="val"
    )
    agent.arun.assert_awaited_once_with("task-x", "extra_arg", custom_kw="val")


@pytest.mark.asyncio
async def test_run_agents_concurrently_async_return_exceptions():
    class FailingAgent:
        async def arun(self, task, *args, **kwargs):
            raise ValueError("failure inside agent")

    res = await run_agents_concurrently_async(
        [FailingAgent()], "task", return_exceptions=True
    )
    assert len(res) == 1
    assert isinstance(res[0], ValueError)
