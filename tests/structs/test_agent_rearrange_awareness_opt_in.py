import pytest
from unittest.mock import MagicMock

from swarms.structs.agent import Agent
from swarms.structs.agent_rearrange import AgentRearrange


def _make_agent(name="Agent-1"):
    agent = Agent.__new__(Agent)
    agent.agent_name = name
    agent.system_prompt = "base system prompt"
    agent.streaming_on = False
    agent.stream = False
    return agent


def test_sequential_workflow_omits_awareness_when_team_awareness_false():
    calls = []

    def stub_run(task=None, messages=None, *args, **kwargs):
        calls.append({"task": task, "messages": messages or []})
        return "result"

    a1 = _make_agent("A")
    a2 = _make_agent("B")
    a1.run = stub_run
    a2.run = stub_run

    rearrange = AgentRearrange(
        agents=[a1, a2], flow="A -> B", max_loops=1, team_awareness=False
    )
    rearrange.run("start task")

    assert len(calls) == 2
    for call in calls:
        system_msgs = [
            m.get("content", "")
            for m in call["messages"]
            if m.get("role") == "system"
        ]
        assert not any("Sequential" in c for c in system_msgs), (
            f"Expected no sequential awareness when team_awareness=False, but found: {system_msgs}"
        )


def test_sequential_workflow_includes_awareness_when_team_awareness_true():
    calls = []

    def stub_run(task=None, messages=None, *args, **kwargs):
        calls.append({"task": task, "messages": messages or []})
        return "result"

    a1 = _make_agent("A")
    a2 = _make_agent("B")
    a1.run = stub_run
    a2.run = stub_run

    rearrange = AgentRearrange(
        agents=[a1, a2], flow="A -> B", max_loops=1, team_awareness=True
    )
    rearrange.run("start task")

    assert len(calls) == 2
    all_system_msgs = []
    for call in calls:
        all_system_msgs.extend(
            m.get("content", "")
            for m in call["messages"]
            if m.get("role") == "system"
        )
    assert any("Sequential" in c for c in all_system_msgs), (
        "Expected sequential awareness when team_awareness=True"
    )


@pytest.mark.asyncio
async def test_stream_sequential_agent_respects_team_awareness_flag():
    a1 = _make_agent("A")
    a2 = _make_agent("B")

    seen_prompts = []

    async def stub_stream(*a, **kw):
        seen_prompts.append(getattr(a1, "system_prompt", ""))
        yield "token"

    a1.arun_stream = stub_stream

    # With team_awareness=False
    rearrange_off = AgentRearrange(
        agents=[a1, a2], flow="A -> B", team_awareness=False
    )
    a1.system_prompt = "base prompt"
    async for _ in rearrange_off._run_sequential_workflow_stream("A", [["A"], ["B"]], task_idx=0):
        pass
    assert not any("Sequential" in p for p in seen_prompts), (
        f"Expected no sequential awareness in stream when team_awareness=False, got {seen_prompts}"
    )

    seen_prompts.clear()
    # With team_awareness=True
    rearrange_on = AgentRearrange(
        agents=[a1, a2], flow="A -> B", team_awareness=True
    )
    a1.system_prompt = "base prompt"
    async for _ in rearrange_on._run_sequential_workflow_stream("A", [["A"], ["B"]], task_idx=0):
        pass
    assert any("Sequential" in p for p in seen_prompts), (
        "Expected sequential awareness in stream when team_awareness=True"
    )
