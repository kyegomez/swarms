"""Test the actual trial example with offline agent and conversation doubles."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


def make_agent(name: str, *testimony: str) -> SimpleNamespace:
    answers = iter(testimony)

    def run(task: str) -> str:
        return (
            next(answers)
            if task.startswith("Provide testimony")
            else name
        )

    return SimpleNamespace(agent_name=name, run=Mock(side_effect=run))


class ConversationDouble:
    def __init__(self) -> None:
        self.conversation_history = []

    def add(self, role: str, content: str) -> None:
        self.conversation_history.append(
            {"role": role, "content": content}
        )


class TestTrialSimulationOffline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        names = (
            "swarms",
            "swarms.structs",
            "swarms.utils",
            "swarms.structs.agent",
            "swarms.structs.conversation",
            "swarms.utils.history_output_formatter",
        )
        modules = {name: ModuleType(name) for name in names}
        modules["swarms.structs.agent"].Agent = object
        modules["swarms.structs.conversation"].Conversation = (
            ConversationDouble
        )
        modules[
            "swarms.utils.history_output_formatter"
        ].history_output_formatter = (
            lambda conversation, type: conversation.conversation_history
        )
        spec = importlib.util.spec_from_file_location(
            __name__ + "_example",
            Path(__file__).with_name("trial_simulation.py"),
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)
        cls.Trial = module.TrialSimulation

    def setUp(self) -> None:
        self.roles = [
            make_agent(name)
            for name in ("Prosecution", "Defense", "Judge")
        ]
        self.witnesses = [
            make_agent("First", "one", "one-next"),
            make_agent("Second", "two", "two-next"),
        ]
        self.trial = self.Trial(
            *self.roles,
            witnesses=self.witnesses,
            phases=["testimony", "cross"],
        )

    def crosses(self) -> list[str]:
        prefix = "Cross-examine this testimony: "
        tasks = [
            call.kwargs["task"]
            for call in self.roles[0].run.call_args_list
        ]
        return [
            task.removeprefix(prefix)
            for task in tasks
            if task.startswith(prefix)
        ]

    def assert_no_calls(self) -> None:
        for agent in self.roles + self.witnesses:
            agent.run.assert_not_called()

    def test_each_witness_keeps_own_testimony(self) -> None:
        self.trial.run("case")
        self.assertEqual(self.crosses(), ["one", "two"])

    def test_cross_only_rejected_before_agent_calls(self) -> None:
        self.trial.phases = ["cross"]
        with self.assertRaisesRegex(ValueError, "[Tt]estimony"):
            self.trial.run("case")
        self.assert_no_calls()

    def test_cross_before_testimony_rejected_before_agent_calls(
        self,
    ) -> None:
        self.trial.phases = ["cross", "testimony"]
        with self.assertRaisesRegex(ValueError, "[Tt]estimony"):
            self.trial.run("case")
        self.assert_no_calls()

    def test_no_witness_cross_allowed(self) -> None:
        for witnesses in (None, []):
            with self.subTest(witnesses=witnesses):
                self.trial.witnesses = witnesses
                self.trial.phases = ["cross"]
                self.trial.run("case")
                self.assertEqual(self.crosses(), [])

    def test_repeated_phases_use_latest_testimony(self) -> None:
        self.trial.phases *= 2
        self.trial.run("case")
        self.assertEqual(
            self.crosses(), ["one", "two", "one-next", "two-next"]
        )

    def test_duplicate_agent_names_do_not_collide(self) -> None:
        self.witnesses[1].agent_name = self.witnesses[0].agent_name
        self.trial.run("case")
        self.assertEqual(self.crosses(), ["one", "two"])

    def test_second_run_uses_new_testimony(self) -> None:
        self.trial.run("first case")
        self.roles[0].run.reset_mock()
        self.trial.run("second case")
        self.assertEqual(self.crosses(), ["one-next", "two-next"])

    def test_second_run_cannot_reuse_previous_testimony(self) -> None:
        self.trial.run("first case")
        for agent in self.roles + self.witnesses:
            agent.run.reset_mock()
        self.trial.phases = ["cross"]
        with self.assertRaisesRegex(ValueError, "[Tt]estimony"):
            self.trial.run("second case")
        self.assert_no_calls()

    def test_empty_testimony_is_recorded(self) -> None:
        self.trial.witnesses = [
            make_agent("Silent", ""),
            self.witnesses[1],
        ]
        self.trial.run("case")
        self.assertEqual(self.crosses(), ["", "two"])


if __name__ == "__main__":
    unittest.main()
