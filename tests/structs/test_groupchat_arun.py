import json
import pytest
from swarms.structs.groupchat import (
    GroupChat,
    RESPOND_TOOL,
)


def respond_call(score, message):
    """A respond() tool call shaped like parsed provider output."""
    return [
        {
            "function": {
                "name": "respond",
                "arguments": json.dumps(
                    {"score": score, "message": message}
                ),
            }
        }
    ]


class AsyncScriptedAgent:
    """Agent implementing native async arun() for testing event loop execution."""

    def __init__(self, name, bids=()):
        self.agent_name = name
        self.tools_list_dictionary = [RESPOND_TOOL]
        self._bids = list(bids)
        self.async_calls = 0
        self.sync_calls = 0

    async def arun(self, task=None, *args, **kwargs):
        self.async_calls += 1
        if self._bids:
            score, message = self._bids.pop(0)
        else:
            score, message = 0.0, ""
        return respond_call(score, message)

    def run(self, task=None, *args, **kwargs):
        self.sync_calls += 1
        if self._bids:
            score, message = self._bids.pop(0)
        else:
            score, message = 0.0, ""
        return respond_call(score, message)


class SyncScriptedAgent:
    """Agent with only synchronous run() method."""

    def __init__(self, name, bids=()):
        self.agent_name = name
        self.tools_list_dictionary = [RESPOND_TOOL]
        self._bids = list(bids)
        self.sync_calls = 0

    def run(self, task=None, *args, **kwargs):
        self.sync_calls += 1
        if self._bids:
            score, message = self._bids.pop(0)
        else:
            score, message = 0.0, ""
        return respond_call(score, message)


@pytest.mark.asyncio
async def test_groupchat_arun_basic():
    agent_a = AsyncScriptedAgent("Agent-A", [(0.9, "Answer from A")])
    agent_b = AsyncScriptedAgent("Agent-B", [(0.4, "Answer from B")])
    chat = GroupChat(agents=[agent_a, agent_b], max_loops=2)

    result = await chat.arun("What is the plan?")
    assert result is not None
    # Agent A had highest bid, so it speaks
    history_roles = [
        msg["role"] for msg in chat.conversation.conversation_history
    ]
    assert "Agent-A" in history_roles


@pytest.mark.asyncio
async def test_groupchat_arun_native_async_bids():
    agent_a = AsyncScriptedAgent("Async-A", [(0.8, "First turn")])
    agent_b = AsyncScriptedAgent("Async-B", [(0.3, "Second turn")])
    chat = GroupChat(agents=[agent_a, agent_b], max_loops=2)

    result = await chat.arun("Start debate")
    assert result is not None
    assert agent_a.async_calls > 0
    assert agent_b.async_calls > 0
    assert agent_a.sync_calls == 0
    assert agent_b.sync_calls == 0


@pytest.mark.asyncio
async def test_groupchat_arun_sync_agent_fallback():
    sync_agent = SyncScriptedAgent("Sync-A", [(0.8, "Sync reply")])
    async_agent = AsyncScriptedAgent("Async-B", [(0.2, "Low bid")])
    chat = GroupChat(agents=[sync_agent, async_agent], max_loops=2)

    result = await chat.arun("Sync fallback test")
    assert result is not None
    assert sync_agent.sync_calls > 0
    assert async_agent.async_calls > 0


@pytest.mark.asyncio
async def test_groupchat_arun_streaming_callback():
    chunks = []

    def callback(agent_name, chunk, is_final):
        chunks.append((agent_name, chunk, is_final))

    agent_1 = AsyncScriptedAgent("Streamer", [(0.9, "Hello streaming world")])
    agent_2 = AsyncScriptedAgent("Listener", [(0.2, "Got it")])
    chat = GroupChat(agents=[agent_1, agent_2], max_loops=1)

    result = await chat.arun("Test stream", streaming_callback=callback)
    assert result is not None
    assert len(chunks) > 0
    assert any(c[2] is True for c in chunks)


@pytest.mark.asyncio
async def test_groupchat_batch_arun():
    agent_1 = AsyncScriptedAgent("Worker1", [(0.8, "Batch reply 1")])
    agent_2 = AsyncScriptedAgent("Worker2", [(0.3, "Batch reply 2")])
    chat = GroupChat(agents=[agent_1, agent_2], max_loops=1)

    tasks = ["Task 1", "Task 2"]
    results = await chat.batch_arun(tasks)
    assert len(results) == 2
    assert all(r is not None for r in results)


@pytest.mark.asyncio
async def test_groupchat_arun_max_concurrency():
    agent_1 = AsyncScriptedAgent("Conc1", [(0.7, "Reply 1")])
    agent_2 = AsyncScriptedAgent("Conc2", [(0.6, "Reply 2")])
    chat = GroupChat(agents=[agent_1, agent_2], max_loops=2, max_concurrency=1)

    result = await chat.arun("Concurrency cap test")
    assert result is not None
    assert agent_1.async_calls > 0
    assert agent_2.async_calls > 0
