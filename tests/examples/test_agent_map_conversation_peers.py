"""Test conversation membership without plotting or model calls."""

import ast
import math
import random
import socket
import sys
import threading
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Optional
from unittest.mock import Mock, patch

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "examples/multi_agent/simulations/agent_map"
    / "agent_map_simulation.py"
)


def load_conversation_state():
    """Load actual state classes without importing the example."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name
        in {"Position", "AgentState", "ConversationManager"}
    ]
    module = ModuleType("offline_agent_map_state")
    module.__dict__.update(
        Agent=object,
        Any=Any,
        Dict=dict,
        List=list,
        Optional=Optional,
        dataclass=dataclass,
        math=math,
        random=random,
        threading=threading,
        time=time,
        logger=Mock(),
        ThreadPoolExecutor=Mock(),
    )
    with patch.dict(sys.modules, {module.__name__: module}):
        exec(  # noqa: S102 - only local example classes
            compile(
                ast.Module(body=classes, type_ignores=[]),
                str(SOURCE),
                "exec",
            ),
            module.__dict__,
        )
    return module


class ConversationPeerTests(unittest.TestCase):
    def setUp(self):
        network = patch.object(
            socket,
            "socket",
            side_effect=AssertionError("Network access is forbidden"),
        )
        network.start()
        self.addCleanup(network.stop)
        self.module = load_conversation_state()
        self.manager = self.module.ConversationManager()
        self.agents = [
            self.module.AgentState(
                SimpleNamespace(agent_name=name),
                self.module.Position(index * 3, index * 6),
            )
            for index, name in enumerate("ABCD")
        ]
        self.conversation_id = self.manager.start_conversation(
            *self.agents[:2], topic="Offline fixture"
        )
        self.conversation = self.manager.active_conversations[
            self.conversation_id
        ]
        self.conversation["status"] = "active"

    def join(self, index):
        return self.manager.try_join_conversation(
            self.agents[index], self.conversation_id
        )

    def assert_peers(self, count):
        names = [
            agent.agent.agent_name for agent in self.agents[:count]
        ]
        for agent in self.agents[:count]:
            expected = [
                name
                for name in names
                if name != agent.agent.agent_name
            ]
            with self.subTest(agent=agent.agent.agent_name):
                self.assertEqual(
                    agent.conversation_partners, expected
                )
                self.assertEqual(
                    agent.conversation_partner, expected[0]
                )

    def test_initial_pair_has_reciprocal_peers(self):
        self.assert_peers(2)

    def test_third_member_updates_existing_participants(self):
        self.assertTrue(self.join(2))
        self.assert_peers(3)

    def test_fourth_member_updates_the_whole_group(self):
        self.assertTrue(self.join(2))
        self.assertTrue(self.join(3))
        self.assert_peers(4)

    def test_join_keeps_conversation_flags_and_updates_center(self):
        self.assertTrue(self.join(2))
        self.assertEqual(
            self.conversation["participants"], self.agents[:3]
        )
        self.assertEqual(
            self.conversation["conversation_center"],
            self.module.Position(3, 6),
        )
        for agent in self.agents[:3]:
            self.assertTrue(agent.is_in_conversation)
            self.assertEqual(
                agent.conversation_id, self.conversation_id
            )
        self.manager.executor.submit.assert_called_once()

    def test_full_conversation_rejects_join_without_changes(self):
        self.conversation["max_participants"] = 2
        self.assertFalse(self.join(2))
        self.assert_peers(2)
        self.assertEqual(
            self.conversation["participants"], self.agents[:2]
        )
        self.assertFalse(self.agents[2].is_in_conversation)
        self.assertEqual(self.agents[2].conversation_partners, [])

    def test_inactive_conversation_rejects_join_without_changes(self):
        self.conversation["status"] = "completed"
        self.assertFalse(self.join(2))
        self.assert_peers(2)
        self.assertFalse(self.agents[2].is_in_conversation)

    def test_existing_member_cannot_join_twice(self):
        self.assertFalse(self.join(0))
        self.assert_peers(2)
        self.assertEqual(
            self.conversation["participants"], self.agents[:2]
        )


if __name__ == "__main__":
    unittest.main()
