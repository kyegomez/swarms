import asyncio
import pytest
from swarms.structs.round_robin import RoundRobinSwarm


class _Memory:
    def __init__(self):
        self.last = ""

    def get_final_message_content(self):
        return self.last


class AsyncTestAgent:
    def __init__(self, name: str):
        self.agent_name = name
        self.short_memory = _Memory()
        self.async_calls = 0
        self.sync_calls = 0
        self.calls = []

    async def arun(self, task=None, *args, messages=None, **kwargs):
        self.async_calls += 1
        output = f"{self.agent_name}-async-out"
        self.short_memory.last = output
        self.calls.append({"task": str(task), "messages": messages or []})
        return output

    def run(self, task=None, *args, messages=None, **kwargs):
        self.sync_calls += 1
        output = f"{self.agent_name}-sync-out"
        self.short_memory.last = output
        self.calls.append({"task": str(task), "messages": messages or []})
        return output


class SyncTestAgent:
    def __init__(self, name: str):
        self.agent_name = name
        self.short_memory = _Memory()
        self.sync_calls = 0
        self.calls = []

    def run(self, task=None, *args, messages=None, **kwargs):
        self.sync_calls += 1
        output = f"{self.agent_name}-sync-out"
        self.short_memory.last = output
        self.calls.append({"task": str(task), "messages": messages or []})
        return output


@pytest.mark.asyncio
async def test_round_robin_arun_basic():
    agents = [AsyncTestAgent(f"A{i}") for i in range(3)]
    swarm = RoundRobinSwarm(agents=agents, max_loops=1, output_type="dict")

    result = await swarm.arun("Initial task")

    assert [m["role"] for m in result] == ["User", "A0", "A1", "A2"]
    for agent in agents:
        assert agent.async_calls == 1
        assert agent.sync_calls == 0


@pytest.mark.asyncio
async def test_round_robin_arun_multi_loop():
    agents = [AsyncTestAgent("A0"), AsyncTestAgent("A1")]
    swarm = RoundRobinSwarm(agents=agents, max_loops=2, output_type="dict")

    result = await swarm.arun("Two loop deliberation")

    expected_roles = ["User", "A0", "A1", "A0", "A1"]
    assert [m["role"] for m in result] == expected_roles
    assert agents[0].async_calls == 2
    assert agents[1].async_calls == 2


@pytest.mark.asyncio
async def test_round_robin_arun_sync_agent_fallback():
    agents = [SyncTestAgent("Sync0"), AsyncTestAgent("Async1")]
    swarm = RoundRobinSwarm(agents=agents, max_loops=1, output_type="dict")

    result = await swarm.arun("Sync fallback check")

    assert [m["role"] for m in result] == ["User", "Sync0", "Async1"]
    assert agents[0].sync_calls == 1
    assert agents[1].async_calls == 1


@pytest.mark.asyncio
async def test_round_robin_batch_arun():
    agents = [AsyncTestAgent("Worker0"), AsyncTestAgent("Worker1")]
    swarm = RoundRobinSwarm(agents=agents, max_loops=1, output_type="dict")

    tasks = ["Task Alpha", "Task Beta"]
    results = await swarm.batch_arun(tasks)

    assert len(results) == 2
    for res in results:
        assert [m["role"] for m in res] == ["User", "Worker0", "Worker1"]


@pytest.mark.asyncio
async def test_round_robin_arun_returns_final_output_by_default():
    agents = [AsyncTestAgent("First"), AsyncTestAgent("Last")]
    swarm = RoundRobinSwarm(agents=agents, max_loops=1, output_type="final")

    result = await swarm.arun("Check final output")
    assert result == "Last-async-out"
