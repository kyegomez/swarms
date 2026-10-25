"""Regression tests for the channel seed surviving ``run()``.

Issue: GroupChat, HierarchicalSwarm and PlannerWorkerSwarm seed
``self.conversation`` from ``channel=`` in their constructors, but every
``run()`` started with ``self.conversation.clear()``, which threw the seed
away. The agents in these structures never saw the turns the channel file
already held, even though the file itself kept them (#2826 says a non-empty
channel makes a run continue from where the file left off).
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from swarms.structs.conversation import Conversation
from swarms.structs.file_channel import FileChannel
from swarms.structs.groupchat import RESPOND_TOOL, GroupChat


class _SeamStop(Exception):
    """Raised at the first LLM-touching seam to stop a run offline."""


class Scripted:
    """A minimal offline agent: bids a fixed score, records what it saw."""

    def __init__(self, name, bids):
        self.agent_name = name
        self.tools_list_dictionary = [RESPOND_TOOL]
        self.bids = list(bids)
        self.seen = []

    def run(self, task=None, messages=None, **kwargs):
        self.seen.append([m["content"] for m in messages or []])
        score, msg = self.bids.pop(0) if self.bids else (0.0, "")
        args = json.dumps({"score": score, "message": msg})
        return [{"function": {"name": "respond", "arguments": args}}]


def _seeded_channel(tmp_path: Path) -> FileChannel:
    channel = FileChannel(Path(tmp_path) / "chat.md")
    channel.post("User", "Pick a database for the ledger.")
    channel.post(
        "Analyst", "Postgres, for its transactional guarantees."
    )
    return channel


def test_groupchat_run_keeps_seeded_channel_turns(tmp_path):
    channel = _seeded_channel(tmp_path)
    analyst = Scripted("Analyst", [(0.9, "Agreed, Postgres.")])
    chat = GroupChat(
        agents=[analyst, Scripted("Critic", [])],
        channel=channel,
        max_loops=3,
        auto_equip=False,
    )

    chat.run("Now pick a cache for it.")

    # The first bid must have seen the turns the channel held before the run.
    assert "Pick a database for the ledger." in analyst.seen[0]
    assert (
        "Postgres, for its transactional guarantees."
        in analyst.seen[0]
    )

    # The run's own history (and the channel file) keep the seed too.
    names = [m["agent_name"] for m in channel.messages()]
    assert names[:2] == ["User", "Analyst"]


def test_groupchat_second_run_continues_the_channel(tmp_path):
    channel = _seeded_channel(tmp_path)
    analyst = Scripted("Analyst", [(0.9, "r1"), (0.9, "r2")])
    chat = GroupChat(
        agents=[analyst, Scripted("Critic", [])],
        channel=channel,
        max_loops=3,
        auto_equip=False,
    )
    chat.run("First run task.")

    analyst2 = Scripted("Analyst", [(0.9, "r3")])
    chat2 = GroupChat(
        agents=[analyst2, Scripted("Critic", [])],
        channel=channel,
        max_loops=3,
        auto_equip=False,
    )
    chat2.run("Second run task.")

    # The second run's agents saw the first run's turns as prior context.
    assert "First run task." in analyst2.seen[0]
    assert "r1" in analyst2.seen[0]


def test_conversation_reseed_does_not_duplicate_channel_rows(
    tmp_path,
):
    channel = FileChannel(tmp_path / "chat.md")
    channel.post("User", "seed one")
    channel.post("Analyst", "seed two")
    conversation = Conversation(time_enabled=False, channel=channel)

    before_rows = list(channel.messages())
    conversation.clear()
    conversation.reseed_from_channel()

    assert [
        m["content"] for m in conversation.conversation_history
    ] == [
        "seed one",
        "seed two",
    ]
    # Reseeding is memory-only: the channel file must not gain rows.
    assert list(channel.messages()) == before_rows


def test_reseed_without_channel_is_a_noop():
    conversation = Conversation(time_enabled=False)
    conversation.clear()
    conversation.reseed_from_channel()  # must not raise


def test_hierarchical_swarm_run_keeps_seeded_turns(tmp_path):
    from swarms.structs.hiearchical_swarm import HierarchicalSwarm

    channel = _seeded_channel(tmp_path)
    swarm = HierarchicalSwarm(
        name="seeded-hierarchy",
        director_name="Director",
        agents=[Scripted("Worker-A", []), Scripted("Worker-B", [])],
        max_loops=1,
        channel=channel,
    )

    with (
        patch.object(
            swarm, "add_context_to_director", side_effect=_SeamStop
        ),
        pytest.raises(_SeamStop),
    ):
        swarm.run("Do the thing.")

    contents = [
        m["content"] for m in swarm.conversation.conversation_history
    ]
    assert "Pick a database for the ledger." in contents
    assert "Postgres, for its transactional guarantees." in contents


def test_planner_worker_swarm_run_keeps_seeded_turns(tmp_path):
    from swarms.structs.planner_worker_swarm import PlannerWorkerSwarm

    channel = _seeded_channel(tmp_path)
    swarm = PlannerWorkerSwarm(
        name="seeded-planner-worker",
        agents=[Scripted("Worker-A", [])],
        max_loops=1,
        channel=channel,
    )

    with (
        patch.object(swarm, "_run_planner", side_effect=_SeamStop),
        pytest.raises(_SeamStop),
    ):
        swarm.run("Do the thing.")

    contents = [
        m["content"] for m in swarm.conversation.conversation_history
    ]
    assert "Pick a database for the ledger." in contents
    assert "Postgres, for its transactional guarantees." in contents
