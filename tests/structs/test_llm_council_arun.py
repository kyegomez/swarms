import asyncio
import time
import pytest
from swarms.structs.llm_council import LLMCouncil


class _Memory:
    def __init__(self):
        self.last = ""

    def get_final_message_content(self):
        return self.last


class AsyncCouncilAgent:
    def __init__(self, name: str, delay: float = 0.0):
        self.agent_name = name
        self.delay = delay
        self.short_memory = _Memory()
        self.async_calls = 0
        self.sync_calls = 0
        self.history = []

    async def arun(self, task=None, *args, **kwargs):
        self.async_calls += 1
        self.history.append(task)
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        output = f"{self.agent_name} answer for {task[:20]}"
        self.short_memory.last = output
        return output

    def run(self, task=None, *args, **kwargs):
        self.sync_calls += 1
        self.history.append(task)
        output = f"{self.agent_name} sync answer for {task[:20]}"
        self.short_memory.last = output
        return output


class SyncCouncilAgent:
    def __init__(self, name: str):
        self.agent_name = name
        self.short_memory = _Memory()
        self.sync_calls = 0
        self.history = []

    def run(self, task=None, *args, **kwargs):
        self.sync_calls += 1
        self.history.append(task)
        output = f"{self.agent_name} sync answer for {task[:20]}"
        self.short_memory.last = output
        return output


@pytest.mark.asyncio
async def test_llm_council_arun_basic():
    members = [AsyncCouncilAgent("MemberA"), AsyncCouncilAgent("MemberB")]
    chairman = AsyncCouncilAgent("Chairman")

    council = LLMCouncil(
        council_members=members,
        verbose=False,
        output_type="dict-all-except-first",
    )
    council.chairman = chairman

    result = await council.arun("Evaluate asynchronous event loop scaling")

    assert result is not None
    history = council.conversation.conversation_history
    roles = [msg["role"] for msg in history]

    assert "User" in roles
    assert "MemberA" in roles
    assert "MemberB" in roles
    assert "MemberA-Evaluation" in roles
    assert "MemberB-Evaluation" in roles
    assert "Chairman" in roles

    assert members[0].async_calls == 2  # 1 response + 1 evaluation
    assert members[1].async_calls == 2
    assert chairman.async_calls == 1
    assert members[0].sync_calls == 0
    assert members[1].sync_calls == 0
    assert chairman.sync_calls == 0


@pytest.mark.asyncio
async def test_llm_council_arun_concurrent_fanout():
    members = [
        AsyncCouncilAgent("M1", delay=0.05),
        AsyncCouncilAgent("M2", delay=0.05),
        AsyncCouncilAgent("M3", delay=0.05),
    ]
    chairman = AsyncCouncilAgent("Chairman", delay=0.01)

    council = LLMCouncil(
        council_members=members,
        verbose=False,
        output_type="dict-all-except-first",
    )
    council.chairman = chairman

    start_time = time.perf_counter()
    result = await council.arun("Concurrency benchmark query")
    elapsed = time.perf_counter() - start_time

    assert result is not None
    # 2 parallel stages (responses + evals) of 0.05s each + 0.01s chairman ~0.11s.
    # Sequential would take (0.05*3) + (0.05*3) + 0.01 = 0.31s+.
    assert elapsed < 0.25


@pytest.mark.asyncio
async def test_llm_council_arun_sync_agent_fallback():
    members = [SyncCouncilAgent("SyncM1"), AsyncCouncilAgent("AsyncM2")]
    chairman = SyncCouncilAgent("SyncChairman")

    council = LLMCouncil(
        council_members=members,
        verbose=False,
        output_type="dict-all-except-first",
    )
    council.chairman = chairman

    result = await council.arun("Fallback test query")

    assert result is not None
    assert members[0].sync_calls == 2
    assert members[1].async_calls == 2
    assert chairman.sync_calls == 1


@pytest.mark.asyncio
async def test_llm_council_batch_arun():
    members = [AsyncCouncilAgent("A"), AsyncCouncilAgent("B")]
    chairman = AsyncCouncilAgent("Chairman")

    council = LLMCouncil(
        council_members=members,
        verbose=False,
        output_type="dict-all-except-first",
    )
    council.chairman = chairman

    tasks = ["Query 1", "Query 2"]
    results = await council.batch_arun(tasks)

    assert len(results) == 2
    for res in results:
        roles = [msg["role"] for msg in res]
        assert "Chairman" in roles


@pytest.mark.asyncio
async def test_llm_council_arun_empty_task_raises():
    members = [AsyncCouncilAgent("A"), AsyncCouncilAgent("B")]
    council = LLMCouncil(council_members=members, verbose=False)

    with pytest.raises(ValueError):
        await council.arun("")

    with pytest.raises(ValueError):
        await council.arun("   ")
