import asyncio
import time
import pytest
from swarms.structs.one_to_three import OneToThree, a_one_to_three, one_to_three


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


def make_three(delay: float = 0.0):
    return [
        AsyncTestAgent("R1", delay=delay),
        AsyncTestAgent("R2", delay=delay),
        AsyncTestAgent("R3", delay=delay),
    ]


@pytest.mark.asyncio
async def test_one_to_three_arun_basic():
    sender = AsyncTestAgent("Sender")
    receivers = make_three()
    swarm = OneToThree(sender, receivers, output_type="dict")

    result = await swarm.arun("Broadcast task")

    assert [m["role"] for m in result] == [
        "User",
        "Sender",
        "R1",
        "R2",
        "R3",
    ]
    assert sender.async_calls == 1
    assert all(r.async_calls == 1 for r in receivers)
    assert all(r.sync_calls == 0 for r in receivers)


@pytest.mark.asyncio
async def test_one_to_three_arun_concurrent_fanout():
    sender = AsyncTestAgent("Sender", delay=0.01)
    receivers = make_three(delay=0.06)
    swarm = OneToThree(sender, receivers, output_type="dict")

    start_time = time.perf_counter()
    result = await swarm.arun("Parallel fanout")
    elapsed = time.perf_counter() - start_time

    assert [m["role"] for m in result] == [
        "User",
        "Sender",
        "R1",
        "R2",
        "R3",
    ]
    # In serial: 0.01 + 3 * 0.06 = 0.19s+. In parallel: ~0.07s.
    assert elapsed < 0.15


@pytest.mark.asyncio
async def test_one_to_three_arun_sync_agent_fallback():
    sender = SyncTestAgent("SyncSender")
    receivers = [
        SyncTestAgent("SyncR1"),
        AsyncTestAgent("AsyncR2"),
        SyncTestAgent("SyncR3"),
    ]
    swarm = OneToThree(sender, receivers, output_type="dict")

    result = await swarm.arun("Mixed sync/async run")

    assert [m["role"] for m in result] == [
        "User",
        "SyncSender",
        "SyncR1",
        "AsyncR2",
        "SyncR3",
    ]
    assert sender.sync_calls == 1
    assert receivers[0].sync_calls == 1
    assert receivers[1].async_calls == 1
    assert receivers[2].sync_calls == 1


@pytest.mark.parametrize("count", [0, 2, 4])
@pytest.mark.asyncio
async def test_one_to_three_arun_requires_exactly_three(count):
    sender = AsyncTestAgent("Sender")
    invalid_receivers = [AsyncTestAgent(f"R{i}") for i in range(count)]

    with pytest.raises(ValueError):
        OneToThree(sender, invalid_receivers)

    with pytest.raises(ValueError):
        await a_one_to_three(sender, invalid_receivers, "task")


@pytest.mark.asyncio
async def test_one_to_three_batch_arun():
    sender = AsyncTestAgent("Sender")
    receivers = make_three()
    swarm = OneToThree(sender, receivers, output_type="dict")

    tasks = ["Batch 1", "Batch 2"]
    results = await swarm.batch_arun(tasks)

    assert len(results) == 2
    for res in results:
        assert [m["role"] for m in res] == ["User", "Sender", "R1", "R2", "R3"]


@pytest.mark.asyncio
async def test_a_one_to_three_standalone():
    sender = AsyncTestAgent("StandaloneSender")
    receivers = make_three()

    result = await a_one_to_three(sender, receivers, "Direct msg", output_type="dict")
    assert [m["role"] for m in result] == ["User", "StandaloneSender", "R1", "R2", "R3"]
