import asyncio
import pytest
from swarms.structs.one_to_one import OneToOne, a_one_to_one, one_to_one


class AsyncTestAgent:
    def __init__(self, name: str):
        self.agent_name = name
        self.async_calls = 0
        self.sync_calls = 0
        self.history = []

    async def arun(self, task=None, *args, **kwargs):
        self.async_calls += 1
        self.history.append({"task": task, "kwargs": kwargs})
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
async def test_one_to_one_arun_basic():
    sender = AsyncTestAgent("Sender")
    receiver = AsyncTestAgent("Receiver")
    pair = OneToOne(sender, receiver, output_type="dict")

    result = await pair.arun("Initiate dialogue")

    assert [m["role"] for m in result] == ["User", "Sender", "Receiver"]
    assert sender.async_calls == 1
    assert sender.sync_calls == 0
    assert receiver.async_calls == 1
    assert receiver.sync_calls == 0
    assert "Sender handled: Initiate dialogue" in receiver.history[0]["task"]


@pytest.mark.asyncio
async def test_one_to_one_arun_multi_loop():
    sender = AsyncTestAgent("Sender")
    receiver = AsyncTestAgent("Receiver")
    pair = OneToOne(sender, receiver, output_type="dict")

    result = await pair.arun("Multi round negotiation", max_loops=3)

    expected_roles = [
        "User",
        "Sender",
        "Receiver",
        "Sender",
        "Receiver",
        "Sender",
        "Receiver",
    ]
    assert [m["role"] for m in result] == expected_roles
    assert sender.async_calls == 3
    assert receiver.async_calls == 3

    # On second turn, sender answers receiver's first response
    assert "Receiver handled:" in sender.history[1]["task"]


@pytest.mark.asyncio
async def test_one_to_one_arun_sync_agent_fallback():
    sender = SyncTestAgent("SyncSender")
    receiver = AsyncTestAgent("AsyncReceiver")
    pair = OneToOne(sender, receiver, output_type="dict")

    result = await pair.arun("Hybrid agent test")

    assert [m["role"] for m in result] == ["User", "SyncSender", "AsyncReceiver"]
    assert sender.sync_calls == 1
    assert receiver.async_calls == 1


@pytest.mark.asyncio
async def test_one_to_one_batch_arun():
    sender = AsyncTestAgent("Sender")
    receiver = AsyncTestAgent("Receiver")
    pair = OneToOne(sender, receiver, output_type="dict")

    tasks = ["Proposal A", "Proposal B"]
    results = await pair.batch_arun(tasks)

    assert len(results) == 2
    for res in results:
        assert [m["role"] for m in res] == ["User", "Sender", "Receiver"]


@pytest.mark.asyncio
async def test_one_to_one_empty_inputs_validation():
    sender = AsyncTestAgent("Sender")
    receiver = AsyncTestAgent("Receiver")
    pair = OneToOne(sender, receiver)

    with pytest.raises(ValueError):
        await pair.arun("")

    with pytest.raises(ValueError):
        await a_one_to_one(None, receiver, "task")

    with pytest.raises(ValueError):
        await a_one_to_one(sender, None, "task")


@pytest.mark.asyncio
async def test_a_one_to_one_standalone_function():
    sender = AsyncTestAgent("SenderA")
    receiver = AsyncTestAgent("ReceiverB")

    result = await a_one_to_one(sender, receiver, "Direct test", max_loops=1, output_type="dict")
    assert [m["role"] for m in result] == ["User", "SenderA", "ReceiverB"]
