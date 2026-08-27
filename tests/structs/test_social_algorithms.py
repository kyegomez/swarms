"""Tests for SocialAlgorithms communication logging (issue #2049).

The logging wrapper used to replace ``Agent.run``/``Agent.talk_to`` on the
class, so it caught every agent in the process and two overlapping instances
left the class permanently wrapped. These tests pin the wrapping to the
agents the structure was handed.
"""

import threading

from swarms.structs.agent import Agent
from swarms.structs.social_algorithms import SocialAlgorithms


class FakeAgent(Agent):
    """An Agent for isinstance checks that never touches an LLM."""

    def __init__(self, name: str = "worker"):
        self.agent_name = name
        self.calls = []

    def run(self, task=None, *args, **kwargs):
        self.calls.append(task)
        return f"answer:{task}"

    def talk_to(self, agent, task, img=None, *args, **kwargs):
        return agent.run(task=f"From {self.agent_name}: {task}")


def build(agents, algorithm):
    return SocialAlgorithms(
        agents=agents,
        social_algorithm=algorithm,
        enable_communication_logging=True,
    )


class TestLoggingIsScopedToOwnedAgents:
    def test_the_agent_class_is_never_patched(self):
        agent = FakeAgent()
        seen = {}

        def algorithm(agents, task, **kwargs):
            seen["run"] = Agent.run
            seen["talk_to"] = Agent.talk_to
            return agents[0].run(task)

        original_run, original_talk_to = Agent.run, Agent.talk_to
        build([agent], algorithm)._wrap_algorithm_with_logging()(
            [agent], "task"
        )

        assert seen["run"] is original_run
        assert seen["talk_to"] is original_talk_to
        assert Agent.run is original_run
        assert Agent.talk_to is original_talk_to

    def test_an_unowned_agent_is_not_logged(self):
        owned, bystander = FakeAgent("owned"), FakeAgent("bystander")
        structure = build([owned], lambda agents, task, **kw: None)

        wrapped = structure._wrap_algorithm_with_logging()

        def algorithm(agents, task, **kwargs):
            bystander.run("a secret from another swarm")
            return agents[0].run(task)

        structure.social_algorithm = algorithm
        wrapped([owned], "owned task")

        logged = [
            step.message for step in structure.communication_history
        ]
        assert logged == ["owned task"]

    def test_owned_agent_calls_are_logged(self):
        agent = FakeAgent("worker")
        structure = build(
            [agent], lambda agents, task, **kw: agents[0].run(task)
        )

        structure._wrap_algorithm_with_logging()([agent], "task")

        assert len(structure.communication_history) == 1
        step = structure.communication_history[0]
        assert step.sender_agent == "worker"
        assert step.receiver_agent == "worker"
        assert step.message == "task"


class TestRestore:
    def test_the_wrapper_is_removed_when_the_algorithm_returns(self):
        agent = FakeAgent()
        structure = build(
            [agent], lambda agents, task, **kw: agents[0].run(task)
        )

        structure._wrap_algorithm_with_logging()([agent], "task")

        assert "run" not in agent.__dict__
        assert "talk_to" not in agent.__dict__

    def test_the_wrapper_is_removed_when_the_algorithm_raises(self):
        agent = FakeAgent()

        def algorithm(agents, task, **kwargs):
            raise RuntimeError("boom")

        structure = build([agent], algorithm)

        try:
            structure._wrap_algorithm_with_logging()([agent], "task")
        except RuntimeError:
            pass

        assert "run" not in agent.__dict__

    def test_a_pre_existing_instance_override_survives(self):
        agent = FakeAgent()

        def override(task=None, **kwargs):
            return "override"

        agent.run = override

        structure = build(
            [agent], lambda agents, task, **kw: agents[0].run(task)
        )
        structure._wrap_algorithm_with_logging()([agent], "task")

        assert agent.run is override

    def test_a_repeated_agent_is_patched_once(self):
        agent = FakeAgent()
        structure = build(
            [agent], lambda agents, task, **kw: agents[0].run(task)
        )

        structure._wrap_algorithm_with_logging()(
            [agent, agent], "task"
        )

        assert len(structure.communication_history) == 1
        assert "run" not in agent.__dict__


class TestConcurrentInstances:
    def test_two_overlapping_instances_stay_independent(self):
        a, b = FakeAgent("a"), FakeAgent("b")
        both_patched = threading.Barrier(2, timeout=5)

        def algorithm(agents, task, **kwargs):
            both_patched.wait()
            return agents[0].run(task)

        first = build([a], algorithm)
        second = build([b], algorithm)

        threads = [
            threading.Thread(
                target=lambda s=s, ag=ag: s._wrap_algorithm_with_logging()(
                    [ag], f"{ag.agent_name} task"
                ),
                daemon=True,
            )
            for s, ag in ((first, a), (second, b))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert [
            step.message for step in first.communication_history
        ] == ["a task"]
        assert [
            step.message for step in second.communication_history
        ] == ["b task"]

    def test_the_class_is_left_clean_after_overlapping_runs(self):
        a, b = FakeAgent("a"), FakeAgent("b")
        original_run = Agent.run
        both_patched = threading.Barrier(2, timeout=5)

        def algorithm(agents, task, **kwargs):
            both_patched.wait()
            return agents[0].run(task)

        threads = [
            threading.Thread(
                target=lambda ag=ag: build(
                    [ag], algorithm
                )._wrap_algorithm_with_logging()([ag], "task"),
                daemon=True,
            )
            for ag in (a, b)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert Agent.run is original_run
