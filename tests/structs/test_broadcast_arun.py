import asyncio
import time
import pytest
from swarms.structs.broadcast import Broadcast, a_broadcast, broadcast


class AsyncTestAgent:
    def __init__(self, name: str, delay: float = 0.0):
        self.agent_name = name
        self.delay = delay
        self.async_calls = 0
        self.sync_calls = 0
        self.history = []

    async def arun(self, task=None, *args, **kwargs):
        self.async_calls += 1
        self.history.append({"task": task, "kwargs": kwargs})
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        return f"{self.agent_name} handled: {task}"

    def run(self, task=None, *args, **kwargs):
        self.sync_calls += 1
        self.history.append({"task": task, "kwargs": kwargs})
        return f"{self.agent_name} handled sync: {task}"


class SyncTestAgent:
    def __init__(self, name: str):
        self.agent_name = name
        self.sync_calls = 0
        self.history = []

    def run(self, task=None, *args, **kwargs):
        self.sync_calls += 1
        self.history.append({"task": task, "kwargs": kwargs})
        return f"{self.agent_name} handled sync: {task}"


@pytest.mark.asyncio
async def test_broadcast_arun_basic():
    sender = AsyncTestAgent("Sender")
    receivers = [AsyncTestAgent("R1"), AsyncTestAgent("R2")]
    group = Broadcast(sender, receivers, output_type="dict")

    result = await group.arun("System announcement")

    assert [m["role"] for m in result] == ["User", "Sender", "R1", "R2"]
    assert sender.async_calls == 1
    assert sender.sync_calls == 0
    assert receivers[0].async_calls == 1
    assert receivers[0].sync_calls == 0
    assert receivers[1].async_calls == 1
    assert receivers[1].sync_calls == 0


@pytest.mark.asyncio
async def test_broadcast_arun_concurrent_fanout():
    sender = AsyncTestAgent("Sender", delay=0.01)
    # 3 receivers each taking 0.06s. In serial this takes 0.18s+, in parallel ~0.06s.
    receivers = [
        AsyncTestAgent("R1", delay=0.06),
        AsyncTestAgent("R2", delay=0.06),
        AsyncTestAgent("R3", delay=0.06),
    ]
    group = Broadcast(sender, receivers, output_type="dict")

    start_time = time.perf_counter()
    result = await group.arun("Parallel fanout task")
    elapsed = time.perf_counter() - start_time

    assert [m["role"] for m in result] == ["User", "Sender", "R1", "R2", "R3"]
    # Should complete much faster than sequential (0.18s)
    assert elapsed < 0.15


@pytest.mark.asyncio
async def test_broadcast_arun_sync_agent_fallback():
    sender = SyncTestAgent("SyncSender")
    receivers = [
        SyncTestAgent("SyncR1"),
        AsyncTestAgent("AsyncR2"),
    ]
    group = Broadcast(sender, receivers, output_type="dict")

    result = await group.arun("Mixed agent broadcast")

    assert [m["role"] for m in result] == ["User", "SyncSender", "SyncR1", "AsyncR2"]
    assert sender.sync_calls == 1
    assert receivers[0].sync_calls == 1
    assert receivers[1].async_calls == 1


@pytest.mark.asyncio
async def test_broadcast_arun_max_concurrency():
    active_count = 0
    max_active = 0

    class ConcurrencyTrackingAgent:
        def __init__(self, name: str):
            self.agent_name = name

        async def arun(self, task=None, *args, **kwargs):
            nonlocal active_count, max_active
            active_count += 1
            if active_count > max_active:
                max_active = active_count
            await asyncio.sleep(0.02)
            active_count -= 1
            return f"{self.agent_name} done"

    sender = ConcurrencyTrackingAgent("Sender")
    receivers = [ConcurrencyTrackingAgent(f"R{i}") for i in range(5)]
    group = Broadcast(sender, receivers, output_type="dict", max_concurrency=2)

    result = await group.arun("Concurrency cap test")
    assert len(result) == 7
    # Receivers concurrency should have been bounded by 2
    assert max_active <= 2


@pytest.mark.asyncio
async def test_broadcast_batch_arun():
    sender = AsyncTestAgent("Sender")
    receivers = [AsyncTestAgent("R1"), AsyncTestAgent("R2")]
    group = Broadcast(sender, receivers, output_type="dict")

    tasks = ["Task Alpha", "Task Beta"]
    results = await group.batch_arun(tasks)

    assert len(results) == 2
    for res in results:
        assert [m["role"] for m in res] == ["User", "Sender", "R1", "R2"]


@pytest.mark.asyncio
async def test_broadcast_empty_inputs_validation():
    sender = AsyncTestAgent("Sender")
    receivers = [AsyncTestAgent("R1")]
    group = Broadcast(sender, receivers)

    with pytest.raises(ValueError):
        await group.arun("")

    with pytest.raises(ValueError):
        await a_broadcast(None, receivers, "task")

    with pytest.raises(ValueError):
        await a_broadcast(sender, [], "task")


@pytest.mark.asyncio
async def test_a_broadcast_standalone_function():
    sender = AsyncTestAgent("DirectSender")
    receivers = [AsyncTestAgent("DirectR1")]

    result = await a_broadcast(sender, receivers, "Direct msg", output_type="dict")
    assert [m["role"] for m in result] == ["User", "DirectSender", "DirectR1"]
