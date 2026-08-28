"""Context delivery for ``AgentJudge``."""

import pytest

from swarms.agents.agent_judge import AgentJudge
from swarms.structs.agent import Agent


def _judge(max_loops=1):
    """A judge whose LLM answers locally, recording what it received."""
    calls = []
    judge = AgentJudge(
        agent_name="Judge", model_name="gpt-5.4", max_loops=max_loops
    )

    def fake_call_llm(self, task=None, *args, **kwargs):
        calls.append(
            {"task": task, "messages": kwargs.get("messages") or []}
        )
        return f"verdict-{len(calls)}"

    Agent.call_llm = fake_call_llm
    return judge, calls


def test_the_judge_reads_its_own_prior_verdict_as_assistant():
    """A previous evaluation is the judge's own turn, not more material."""
    judge, calls = _judge(max_loops=2)
    judge.run(task="rate this answer")

    assert len(calls) == 2

    own = [
        m
        for m in calls[1]["messages"]
        if str(m["content"]).startswith("verdict-")
    ]
    assert len(own) == 1, f"expected one prior verdict: {own}"
    assert own[0]["role"] == "assistant"


def test_context_does_not_compound_across_loops():
    """Each loop must be a strict prefix of the next, not a re-flattening."""
    judge, calls = _judge(max_loops=3)
    judge.run(task="rate this answer")

    # call_llm receives the whole transcript: the prior turns plus the
    # instruction appended last. The cacheable part is everything before it.
    priors = [
        [m["content"] for m in call["messages"][:-1]]
        for call in calls
    ]

    assert priors[0] == []
    for earlier, later in zip(priors, priors[1:]):
        assert (
            later[: len(earlier)] == earlier
        ), f"loop context was rebuilt rather than extended: {priors}"
        assert len(later) == len(earlier) + 1


def test_the_conversation_is_not_flattened_into_the_task():
    """The old form passed the whole transcript as one user string."""
    judge, calls = _judge(max_loops=2)
    judge.run(task="rate this answer")

    for call in calls:
        assert "Judge:" not in str(
            call["task"]
        ), f"the judge received flattened prose: {call['task']}"


def test_a_single_loop_sends_only_the_instruction():
    judge, calls = _judge(max_loops=1)
    judge.run(task="rate this answer")

    assert len(calls) == 1
    assert len(calls[0]["messages"]) == 1
    assert "rate this answer" in calls[0]["messages"][0]["content"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
