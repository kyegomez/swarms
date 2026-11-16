"""Offline tests for SelfOrganizingSwarm."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from swarms.structs.self_organize import (
    MessageBoard,
    SelfOrganizingSwarm,
    _Episode,
    _final_answer,
    _output_tokens,
)


class FakeMemory:
    """Stand-in for an agent's short memory."""

    def __init__(self):
        self.entries = []

    def add(self, role=None, content=None, **kwargs):
        self.entries.append({"role": role, "content": content})


class FakeAgent:
    """Scripted agent that replays one reply queue per member."""

    queues: list = []
    _next = 0

    def __init__(self, **kwargs):
        self.agent_name = kwargs.get("agent_name")
        self.stopping_func = kwargs.get("stopping_func")
        self.tools = kwargs.get("tools") or []
        self.max_loops = kwargs.get("max_loops", 1)
        self.system_prompt = kwargs.get("system_prompt", "")
        self.usage = {"output_tokens": 10}
        self.short_memory = FakeMemory()
        self.calls = []

        idx = FakeAgent._next
        FakeAgent._next += 1
        if idx < len(FakeAgent.queues):
            self._queue = list(FakeAgent.queues[idx])
        else:
            self._queue = ["#### ok"]

    def run(self, task=None, *args, **kwargs):
        self.calls.append(task)
        text = self._queue.pop(0) if self._queue else "#### ok"
        if self.stopping_func is not None:
            self.stopping_func(text)
        return text


@pytest.fixture
def patch_agent(monkeypatch):
    """Replace Agent with FakeAgent."""
    FakeAgent.queues = []
    FakeAgent._next = 0
    monkeypatch.setattr(
        "swarms.structs.self_organize.Agent", FakeAgent
    )
    return FakeAgent


def make_swarm(**kwargs) -> SelfOrganizingSwarm:
    kwargs.setdefault("num_agents", 2)
    kwargs.setdefault("seed", 0)
    kwargs.setdefault("timeout", 5.0)
    kwargs.setdefault("max_tokens_per_agent", 1024)
    kwargs.setdefault("token_cost", 0.25)
    return SelfOrganizingSwarm(**kwargs)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("#### 42", "42"),
        ("thinking...\n#### [1, 2]\n", "[1, 2]"),
        ("#### first\n#### second", "second"),
        ("no marker here", None),
        ("", None),
        (None, None),
    ],
)
def test_final_answer_parsing(text, expected):
    assert _final_answer(text) == expected


def test_output_tokens_reads_usage():
    agent = SimpleNamespace(usage={"output_tokens": 99})
    assert _output_tokens(agent) == 99


def test_output_tokens_missing_usage_returns_zero():
    assert _output_tokens(SimpleNamespace()) == 0
    assert _output_tokens(SimpleNamespace(usage={})) == 0


@pytest.fixture
def board():
    return MessageBoard(["a", "b", "c"], threading.Condition())


def test_message_board_direct_delivery(board):
    status = board.post("a", "b", "hello")
    assert "b" in status

    msgs = board.read("b")
    assert len(msgs) == 1
    assert msgs[0]["from"] == "a"
    assert msgs[0]["to"] == "b"
    assert msgs[0]["text"] == "hello"
    assert board.read("b") == []
    assert board.read("c") == []


def test_message_board_broadcast_skips_sender(board):
    board.post("a", "all", "ping")
    assert len(board.read("b")) == 1
    assert len(board.read("c")) == 1
    assert board.read("a") == []


def test_message_board_rejects_unknown_and_self(board):
    assert "Unknown recipient" in board.post("a", "z", "x")
    assert "Unknown recipient" in board.post("a", "a", "x")


def test_message_board_unread_counts_and_notification(board):
    board.post("a", "b", "one")
    board.post("c", "b", "two")
    board.post("a", "b", "three")

    assert board.unread_counts("b") == {"a": 2, "c": 1}

    note = json.loads(board.notification("b"))
    assert note["unread_by_sender"] == {"a": 2, "c": 1}
    assert "read_messages" in note["notice"]
    assert board.notification("b") is None
    assert len(board.read("b")) == 3


def test_message_board_log_records_every_post(board):
    board.post("a", "b", "x")
    board.post("b", "all", "y")
    assert len(board.log) == 2
    assert board.log[1]["to"] == "all"


def test_episode_submit_with_verifier_stops_on_success():
    ep = _Episode(
        ["a", "b"],
        verifier=lambda ans: ans == "ok",
        stop_on_success=True,
        timeout=30.0,
        verbose=False,
    )
    ep.submit("a", "wrong")
    assert ep.solved_by is None
    assert ep.phase == "work"
    assert ep.answers["a"] == "wrong"

    ep.submit("b", "ok")
    assert ep.solved_by == "b"
    assert ep.solution == "ok"
    assert ep.phase == "done"


def test_episode_solve_from_environment():
    ep = _Episode(
        ["a"],
        verifier=None,
        stop_on_success=True,
        timeout=30.0,
        verbose=False,
    )
    ep.solve("a", "exit 7")
    assert ep.env_solved is True
    assert ep.solved_by == "a"
    assert ep.solution == "exit 7"
    assert ep.phase == "done"


def test_episode_quiet_when_everyone_idle():
    ep = _Episode(
        ["a", "b"],
        verifier=None,
        stop_on_success=True,
        timeout=30.0,
        verbose=False,
    )
    assert ep.quiet() is False
    ep.status["a"] = "idle"
    ep.status["b"] = "idle"
    assert ep.quiet() is True

    ep.board.pending["a"] = True
    assert ep.quiet() is False


def test_rejects_zero_agents():
    with pytest.raises(ValueError, match="at least 1"):
        SelfOrganizingSwarm(num_agents=0)


def test_rejects_mismatched_private_inputs(patch_agent):
    swarm = make_swarm(num_agents=2)
    with pytest.raises(ValueError, match="private inputs"):
        swarm.run("task", private_inputs=["only-one"])


def test_system_prompt_includes_messaging_when_enabled():
    swarm = make_swarm(communication=True)
    prompt = swarm._system_prompt(
        "agent_1", ["agent_2"], "Share facts."
    )
    assert "agent_1" in prompt
    assert "agent_2" in prompt
    assert "message(to, text)" in prompt
    assert "Share facts." in prompt


def test_system_prompt_solo_mode_omits_messaging():
    swarm = make_swarm(communication=False)
    prompt = swarm._system_prompt("agent_1", [], "")
    assert "independently" in prompt.lower()
    assert "message(to, text)" not in prompt


def test_run_verified_answer_and_team_reward(patch_agent):
    FakeAgent.queues = [["#### 42"], ["still thinking"]]
    swarm = make_swarm(num_agents=2, token_cost=0.5)

    result = swarm.run(
        "What is the answer?",
        verifier=lambda a: a.strip() == "42",
    )

    assert result["success"] is True
    assert result["answer"] == "42"
    assert result["solved_by"] is not None
    assert result["team_reward"] is not None
    assert 0.0 < result["team_reward"] <= 1.0
    assert result["communication"] is True
    assert len(result["agents"]) == 2
    assert "elapsed_seconds" in result
    assert result["total_tokens"] >= 0


def test_run_majority_answer_without_verifier(patch_agent):
    FakeAgent.queues = [
        ["#### red"],
        ["#### blue"],
        ["#### red"],
    ]
    swarm = make_swarm(num_agents=3, seed=1)

    result = swarm.run("Pick a color.")

    assert result["success"] is None
    assert result["answer"] == "red"
    assert result["team_reward"] is None
    assert set(result["answers"].values()) == {"red", "blue"}


def test_run_final_phase_asks_agents_without_answers(patch_agent):
    # First reply has no #### → idle. Quiet room → final phase.
    # Final phase accepts the last non-empty line.
    FakeAgent.queues = [
        ["working...", "17"],
        ["also working...", "17"],
    ]
    swarm = make_swarm(num_agents=2, seed=2)

    result = swarm.run(
        "Final number?",
        verifier=lambda a: a.strip() == "17",
        final_prompt="Submit now.",
    )

    assert result["success"] is True
    assert result["answer"] == "17"


def test_run_task_tools_solve_ends_episode(patch_agent, monkeypatch):
    class ToolCallingAgent(FakeAgent):
        def run(self, task=None, *args, **kwargs):
            self.calls.append(task)
            for tool in self.tools or []:
                if getattr(tool, "__name__", "") == "claim_exit":
                    tool()
                    break
            text = "still exploring"
            if self.stopping_func is not None:
                self.stopping_func(text)
            return text

    FakeAgent.queues = [["unused"]]
    FakeAgent._next = 0
    monkeypatch.setattr(
        "swarms.structs.self_organize.Agent", ToolCallingAgent
    )

    def task_tools(agent_name, solve):
        def claim_exit() -> str:
            """Mark the environment as solved."""
            solve(f"exit-by-{agent_name}")
            return "found exit"

        return [claim_exit]

    swarm = make_swarm(num_agents=1, seed=3, communication=False)
    result = swarm.run(
        "Find the exit.",
        task_tools=task_tools,
        verifier=lambda _: False,
        final_prompt=None,
    )

    assert result["success"] is True
    assert result["solved_by"] is not None
    assert result["answer"].startswith("exit-by-")


def test_run_communication_false_has_empty_message_log(patch_agent):
    FakeAgent.queues = [["#### solo"], ["#### solo"]]
    swarm = make_swarm(num_agents=2, seed=4, communication=False)

    result = swarm.run("Work alone.", verifier=lambda a: a == "solo")

    assert result["communication"] is False
    assert result["message_calls"] == 0
    assert result["messages"] == []
    assert result["success"] is True


def test_messaging_tools_post_and_read(patch_agent):
    """Messages sent by one member reach the other's inbox."""
    swarm = make_swarm(num_agents=2, seed=5)
    ep = _Episode(
        ["alice", "bob"],
        verifier=None,
        stop_on_success=True,
        timeout=30.0,
        verbose=False,
    )
    from swarms.structs.self_organize import _Member

    alice = _Member("alice")
    bob = _Member("bob")
    alice_tools = {
        t.__name__: t for t in swarm._messaging_tools(ep, alice)
    }
    bob_tools = {
        t.__name__: t for t in swarm._messaging_tools(ep, bob)
    }

    status = alice_tools["message"](to="bob", text="secret")
    assert "bob" in status

    counts = json.loads(bob_tools["check_notifications"]())
    assert counts["unread_by_sender"] == {"alice": 1}

    inbox = json.loads(bob_tools["read_messages"]())
    assert inbox == [{"from": "alice", "to": "bob", "text": "secret"}]
    assert bob_tools["read_messages"]() == "No unread messages."


def test_private_inputs_appended_to_first_turn(patch_agent):
    FakeAgent.queues = [["#### done"], ["#### done"]]
    built = []

    swarm = make_swarm(num_agents=2, seed=6)
    original = swarm._build_agent

    def tracking_build(member, names, ep, task_tools, rules):
        agent = original(member, names, ep, task_tools, rules)
        built.append(agent)
        return agent

    swarm._build_agent = tracking_build
    swarm.run(
        "Base task",
        private_inputs=["priv-A", "priv-B"],
        verifier=lambda a: a == "done",
    )

    first_tasks = [agent.calls[0] for agent in built]
    assert any("priv-A" in t for t in first_tasks)
    assert any("priv-B" in t for t in first_tasks)
    assert all(t.startswith("Base task") for t in first_tasks)
