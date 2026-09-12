import asyncio
import threading
import time

import pytest

from swarms.structs.agent import Agent
from swarms.structs.social_algorithms import (
    AgentNotFoundError,
    InvalidAlgorithmError,
    SocialAlgorithms,
)


class FakeAgent(Agent):
    """An Agent that answers locally, so tests need no model or API key."""

    def __init__(self, name):
        self.agent_name = name

    def run(self, task, *args, **kwargs):
        return f"{self.agent_name} handled: {task}"

    def talk_to(self, other, task, *args, **kwargs):
        return other.run(task)


def _agent(name):
    agent = Agent.__new__(Agent)
    agent.agent_name = name
    return agent


def _bare(agent_names):
    """Build an instance without __init__, for cases __init__ would reject."""
    social_algorithm = SocialAlgorithms.__new__(SocialAlgorithms)
    social_algorithm.agents = [_agent(name) for name in agent_names]
    social_algorithm.verbose = False
    return social_algorithm


def _swarm(algorithm, names=("Researcher", "Analyst"), **kwargs):
    return SocialAlgorithms(
        name="Demo",
        agents=[FakeAgent(name) for name in names],
        social_algorithm=algorithm,
        **kwargs,
    )


def _roles(swarm):
    return [
        message["role"]
        for message in swarm.conversation.conversation_history
    ]


def _pipeline(agents, task, **kwargs):
    research = agents[0].run(f"Research: {task}")
    return agents[1].run(f"Analyze: {research}")


class TestValidation:
    def test_empty_agent_list_is_rejected(self):
        with pytest.raises(ValueError, match="At least one agent"):
            SocialAlgorithms(agents=[], social_algorithm=len)

    def test_non_agent_members_are_rejected(self):
        with pytest.raises(
            ValueError, match="instances of the Agent"
        ):
            SocialAlgorithms(
                agents=["not-an-agent"], social_algorithm=len
            )

    def test_non_callable_algorithm_is_rejected(self):
        with pytest.raises(InvalidAlgorithmError):
            SocialAlgorithms(
                agents=[FakeAgent("A")], social_algorithm="nope"
            )

    def test_non_positive_max_execution_time_is_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            SocialAlgorithms(
                agents=[FakeAgent("A")],
                social_algorithm=len,
                max_execution_time=0,
            )

    def test_run_without_an_algorithm_is_rejected(self):
        swarm = SocialAlgorithms(agents=[FakeAgent("A")])

        with pytest.raises(InvalidAlgorithmError):
            swarm.run("t")

    def test_unknown_constructor_kwargs_are_tolerated(self):
        """Retired parameters must not break callers that still pass them."""
        swarm = _swarm(
            _pipeline,
            enable_communication_logging=False,
            parallel_execution=True,
        )

        assert swarm.run("t").total_steps == 2


class TestAgentRoster:
    def test_add_agent_appends(self):
        swarm = _swarm(_pipeline)

        swarm.add_agent(FakeAgent("Writer"))

        assert [a.agent_name for a in swarm.agents] == [
            "Researcher",
            "Analyst",
            "Writer",
        ]

    def test_add_agent_rejects_non_agents(self):
        swarm = _swarm(_pipeline)

        with pytest.raises(ValueError):
            swarm.add_agent("not-an-agent")

    def test_agent_added_after_construction_is_recorded(self):
        swarm = _swarm(
            lambda agents, task, **kwargs: agents[-1].run(task)
        )
        swarm.add_agent(FakeAgent("Writer"))

        swarm.run("t")

        assert "Writer" in _roles(swarm)

    def test_remove_agent_removes_matching_agent_by_name(self):
        swarm = _bare(["researcher", "critic"])

        swarm.remove_agent("researcher")

        assert [a.agent_name for a in swarm.agents] == ["critic"]

    def test_remove_agent_preserves_remaining_agent_order(self):
        swarm = _bare(["planner", "researcher", "critic", "writer"])

        swarm.remove_agent("critic")

        assert [a.agent_name for a in swarm.agents] == [
            "planner",
            "researcher",
            "writer",
        ]

    def test_remove_agent_raises_agent_not_found_for_unknown_name(
        self,
    ):
        with pytest.raises(AgentNotFoundError):
            _bare(["researcher"]).remove_agent("critic")

    def test_remove_agent_raises_agent_not_found_for_empty_agent_list(
        self,
    ):
        with pytest.raises(AgentNotFoundError):
            _bare([]).remove_agent("researcher")


class TestConversationRecording:
    def test_task_agents_and_result_are_recorded_in_order(self):
        swarm = _swarm(_pipeline)

        swarm.run("AI in healthcare")

        assert _roles(swarm) == [
            "User",
            "Researcher",
            "Analyst",
            "Demo",
        ]

    def test_agent_output_is_the_recorded_content(self):
        swarm = _swarm(_pipeline)

        swarm.run("t")

        contents = [
            m["content"]
            for m in swarm.conversation.conversation_history
        ]
        assert contents[0] == "t"
        assert contents[1] == "Researcher handled: Research: t"
        assert contents[2].startswith("Analyst handled: Analyze:")
        assert contents[3] == contents[2]

    def test_talk_to_records_the_receiver(self):
        swarm = _swarm(
            lambda agents, task, **kwargs: agents[0].talk_to(
                agents[1], task
            )
        )

        swarm.run("question")

        sender = swarm.conversation.conversation_history[1]
        assert sender["role"] == "Researcher"
        assert sender["content"] == "question"

    def test_repeated_agent_calls_are_all_recorded(self):
        def chatty(agents, task, **kwargs):
            for i in range(3):
                agents[0].run(f"{task}-{i}")
            return "done"

        swarm = _swarm(chatty)

        swarm.run("t")

        assert _roles(swarm).count("Researcher") == 3

    def test_conversation_accumulates_across_runs(self):
        swarm = _swarm(_pipeline)

        swarm.run("first")
        swarm.run("second")

        assert len(swarm.conversation.conversation_history) == 8

    def test_clear_communication_history_empties_the_transcript(self):
        swarm = _swarm(_pipeline)
        swarm.run("t")

        swarm.clear_communication_history()

        assert swarm.get_communication_history() == []

    def test_get_communication_history_returns_a_copy(self):
        swarm = _swarm(_pipeline)
        swarm.run("t")

        swarm.get_communication_history().clear()

        assert len(swarm.conversation.conversation_history) == 4

    def test_a_nested_swarm_does_not_hide_calls_from_the_outer_one(
        self,
    ):
        """The inner recorder must chain to the outer one, not replace it."""
        shared = FakeAgent("Shared")
        inner = SocialAlgorithms(
            name="Inner",
            agents=[shared],
            social_algorithm=lambda agents, task, **kwargs: agents[
                0
            ].run(task),
        )
        outer = SocialAlgorithms(
            name="Outer",
            agents=[shared],
            social_algorithm=lambda agents, task, **kwargs: inner.run(
                "nested"
            ),
        )

        outer.run("t")

        assert "Shared" in _roles(outer)


class TestAgentRestoration:
    def test_agents_are_unpatched_after_a_successful_run(self):
        swarm = _swarm(_pipeline)

        swarm.run("t")

        for agent in swarm.agents:
            assert "run" not in agent.__dict__
            assert "talk_to" not in agent.__dict__

    def test_agents_are_unpatched_after_the_algorithm_raises(self):
        def boom(agents, task, **kwargs):
            agents[0].run(task)
            raise RuntimeError("boom")

        swarm = _swarm(boom)

        with pytest.raises(RuntimeError):
            swarm.run("t")

        assert "run" not in swarm.agents[0].__dict__

    def test_agents_still_work_normally_after_a_run(self):
        swarm = _swarm(_pipeline)
        swarm.run("t")

        assert (
            swarm.agents[0].run("ping") == "Researcher handled: ping"
        )

    def test_a_pre_existing_instance_override_is_used_and_restored(
        self,
    ):
        agent = FakeAgent("Custom")
        agent.run = lambda task, *a, **k: f"override:{task}"
        swarm = SocialAlgorithms(
            name="Demo",
            agents=[agent],
            social_algorithm=lambda agents, task, **kwargs: agents[
                0
            ].run(task),
        )

        swarm.run("t")

        assert (
            swarm.conversation.conversation_history[1]["content"]
            == "override:t"
        )
        assert agent.run("x") == "override:x"

    def test_the_other_agents_class_is_untouched(self):
        """Patching is per instance, so bystander agents must be unaffected."""
        bystander = FakeAgent("Bystander")
        swarm = _swarm(_pipeline)

        swarm.run("t")

        assert bystander.run("x") == "Bystander handled: x"
        assert "run" not in bystander.__dict__


class TestOutputFormatting:
    def test_dict_output_wraps_a_non_dict_result(self):
        assert _swarm(lambda a, t, **k: "plain").run(
            "t"
        ).final_outputs == {"result": "plain"}

    def test_dict_output_passes_a_dict_through(self):
        payload = {"a": 1}

        assert (
            _swarm(lambda a, t, **k: payload).run("t").final_outputs
            is payload
        )

    def test_list_output_wraps_a_non_list_result(self):
        swarm = _swarm(lambda a, t, **k: "plain", output_type="list")

        assert swarm.run("t").final_outputs == ["plain"]

    def test_str_output_stringifies(self):
        swarm = _swarm(lambda a, t, **k: 42, output_type="str")

        assert swarm.run("t").final_outputs == "42"

    def test_a_none_result_is_handled(self):
        assert _swarm(lambda a, t, **k: None).run(
            "t"
        ).final_outputs == {"result": None}


class TestResultAccounting:
    def test_total_steps_counts_only_agent_messages(self):
        result = _swarm(_pipeline).run("t")

        assert result.total_steps == 2
        assert result.successful_steps == 2

    def test_step_counts_are_per_run_not_cumulative(self):
        swarm = _swarm(_pipeline)
        swarm.run("first")

        assert swarm.run("second").total_steps == 2

    def test_failed_steps_is_zero_on_success(self):
        assert _swarm(_pipeline).run("t").failed_steps == 0

    def test_execution_time_is_recorded(self):
        assert _swarm(_pipeline).run("t").execution_time >= 0

    def test_algorithm_id_is_carried_into_the_result(self):
        swarm = _swarm(_pipeline)

        assert swarm.run("t").algorithm_id == swarm.algorithm_id

    def test_communication_history_matches_the_conversation(self):
        swarm = _swarm(_pipeline)

        result = swarm.run("t")

        assert (
            result.communication_history
            == swarm.conversation.conversation_history
        )


class TestErrorHandling:
    def test_the_algorithms_exception_propagates(self):
        swarm = _swarm(
            lambda a, t, **k: (_ for _ in ()).throw(KeyError("nope"))
        )

        with pytest.raises(KeyError):
            swarm.run("t")

    def test_no_result_message_is_recorded_on_failure(self):
        def boom(agents, task, **kwargs):
            agents[0].run(task)
            raise RuntimeError("boom")

        swarm = _swarm(boom)

        with pytest.raises(RuntimeError):
            swarm.run("t")

        assert _roles(swarm) == ["User", "Researcher"]


class TestAlgorithmArgs:
    def test_run_does_not_write_kwargs_into_the_callers_dict(self):
        seen = {}

        def algorithm(agents, task, **kwargs):
            seen.clear()
            seen.update(kwargs)
            return "ok"

        swarm = _swarm(algorithm)
        shared = {"depth": 3}

        swarm.run("first", algorithm_args=shared, temperature=0.2)

        assert shared == {"depth": 3}
        assert seen == {"depth": 3, "temperature": 0.2}

        swarm.run("second", algorithm_args=shared)

        assert seen == {"depth": 3}

    def test_kwargs_win_over_algorithm_args_on_conflict(self):
        seen = {}

        def algorithm(agents, task, **kwargs):
            seen.update(kwargs)
            return "ok"

        _swarm(algorithm).run(
            "t", algorithm_args={"depth": 1}, depth=9
        )

        assert seen == {"depth": 9}

    def test_the_task_reaches_the_algorithm(self):
        seen = {}

        def algorithm(agents, task, **kwargs):
            seen["task"] = task
            return "ok"

        _swarm(algorithm).run("the actual task")

        assert seen["task"] == "the actual task"


class TestTimeout:
    """The timeout joins a worker thread, so it works anywhere.

    The previous SIGALRM implementation only worked on the main thread of
    Unix platforms; the three formerly-xfailed tests below pin down the
    portability and precision it lacked.
    """

    def test_a_generous_timeout_does_not_interfere(self):
        swarm = _swarm(_pipeline, max_execution_time=30)

        assert swarm.run("t").total_steps == 2

    def test_disabling_the_timeout_still_runs(self):
        swarm = _swarm(_pipeline)
        swarm.max_execution_time = 0

        assert swarm.run("t").total_steps == 2

    def test_a_sub_second_budget_still_times_out(self):
        def slow(agents, task, **kwargs):
            time.sleep(2)
            return "never"

        swarm = _swarm(slow, max_execution_time=0.2)

        with pytest.raises(TimeoutError):
            swarm.run("t")

    def test_a_timeout_does_not_block_until_the_algorithm_ends(self):
        def slow(agents, task, **kwargs):
            time.sleep(5)
            return "never"

        swarm = _swarm(slow, max_execution_time=0.2)
        started = time.monotonic()

        with pytest.raises(TimeoutError):
            swarm.run("t")

        assert time.monotonic() - started < 4

    def test_run_works_on_a_worker_thread(self):
        swarm = _swarm(_pipeline)
        outcome = {}

        def worker():
            try:
                outcome["steps"] = swarm.run("t").total_steps
            except BaseException as exc:
                outcome["error"] = exc

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        assert outcome.get("steps") == 2

    def test_run_works_under_asyncio_to_thread(self):
        """The default configuration must survive the common async pattern."""
        swarm = _swarm(_pipeline)

        async def main():
            return await asyncio.to_thread(swarm.run, "t")

        assert asyncio.run(main()).total_steps == 2


class TestAlgorithmInfo:
    def test_reports_the_current_roster(self):
        swarm = _swarm(_pipeline)
        swarm.add_agent(FakeAgent("Writer"))

        info = swarm.get_algorithm_info()

        assert info["agent_count"] == 3
        assert info["agent_names"] == [
            "Researcher",
            "Analyst",
            "Writer",
        ]
        assert info["has_algorithm"] is True

    def test_reports_no_algorithm_when_unset(self):
        swarm = SocialAlgorithms(agents=[FakeAgent("A")])

        assert swarm.get_algorithm_info()["has_algorithm"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
