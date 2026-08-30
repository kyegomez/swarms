"""Context delivery for ``ReasoningDuo``."""

import pytest

from swarms.agents.reasoning_duo import ReasoningDuo
from swarms.structs.agent import Agent


def _duo(max_loops=1):
    """A duo whose LLM answers locally, recording what each agent received."""
    calls = []
    duo = ReasoningDuo(
        model_names=["gpt-5.4", "gpt-5.4"], max_loops=max_loops
    )

    def fake_call_llm(self, task=None, *args, **kwargs):
        calls.append(
            {
                "agent": self.agent_name,
                "messages": kwargs.get("messages") or [],
            }
        )
        return f"{self.agent_name}-out"

    Agent.call_llm = fake_call_llm
    return duo, calls


def test_the_two_agents_have_distinct_names():
    """A shared name made both agents the same speaker in the conversation."""
    duo, _ = _duo()

    assert (
        duo.reasoning_agent.agent_name != duo.main_agent.agent_name
    ), "both agents answer to one name, so neither can be attributed"


def test_the_main_agent_sees_the_reasoner_as_a_labelled_turn():
    duo, calls = _duo()
    duo.run("think about it")

    main = [c for c in calls if c["agent"].endswith("-main")][0]
    contents = [m["content"] for m in main["messages"]]

    assert any(
        "-reasoning-out" in c for c in contents
    ), f"the main agent never saw the reasoner: {contents}"
    assert any(
        "think about it" in c for c in contents
    ), f"the main agent never saw the task: {contents}"


def test_each_agent_reads_its_own_output_as_assistant():
    duo, calls = _duo(max_loops=2)
    duo.run("think about it")

    second_reasoning = [
        c for c in calls if c["agent"].endswith("-reasoning")
    ][1]
    own = [
        m
        for m in second_reasoning["messages"]
        if m["content"].endswith("-reasoning-out")
    ]

    assert len(own) == 1, f"expected one own turn: {own}"
    assert own[0]["role"] == "assistant"


def test_the_task_is_not_duplicated_across_the_loop():
    """run() used to seed the task and then step() appended it again."""
    duo, calls = _duo(max_loops=2)
    duo.run("think about it")

    first = calls[0]
    contents = [m["content"] for m in first["messages"]]

    assert contents.count("think about it") == 1, contents


def test_no_agent_receives_a_flattened_transcript():
    """The old form passed conversation.get_str() as the task."""
    duo, calls = _duo(max_loops=2)
    duo.run("think about it")

    for call in calls:
        for message in call["messages"]:
            assert (
                "Previous conversation context:"
                not in message["content"]
            ), f"{call['agent']} received a flattened transcript"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
