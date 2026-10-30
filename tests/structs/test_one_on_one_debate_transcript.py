import pytest
from swarms.structs.agent import Agent
from swarms.structs.multi_agent_debates import OneOnOneDebate


def agent_with_fake_llm(name):
    agent = Agent(
        agent_name=name, model_name="gpt-5.4", max_loops=1, print_on=False
    )
    counter = {"n": 0}

    def call_llm(task, *args, **kwargs):
        counter["n"] += 1
        return f"{name} point {counter['n']}"

    agent.call_llm = call_llm
    return agent


def test_one_on_one_debate_does_not_compound_transcripts():
    """Each turn in OneOnOneDebate should hold only the speaker's point, not full past transcripts."""
    pro = agent_with_fake_llm("Pro")
    con = agent_with_fake_llm("Con")

    debate = OneOnOneDebate(
        max_loops=3,
        agents=[pro, con],
        output_type="list",
    )
    turns = debate.run("Is remote work better?")

    assert len(turns) == 3
    # Turn 0: Pro's response to task (turn 1 was the intro priming)
    assert turns[0]["role"] == "Pro"
    assert turns[0]["content"] == "Pro point 2"

    # Turn 1: Con's response to Pro point 2
    assert turns[1]["role"] == "Con"
    assert turns[1]["content"] == "Con point 2"

    # Turn 2: Pro's response to Con point 2
    assert turns[2]["role"] == "Pro"
    assert turns[2]["content"] == "Pro point 3"


def test_one_on_one_debate_requires_exactly_two_agents():
    """OneOnOneDebate must raise ValueError if agent count != 2."""
    pro = agent_with_fake_llm("Pro")
    debate = OneOnOneDebate(agents=[pro])
    with pytest.raises(ValueError, match="There must be exactly two agents"):
        debate.run("Topic")


def test_one_on_one_debate_string_output():
    """OneOnOneDebate formats string output correctly."""
    pro = agent_with_fake_llm("Pro")
    con = agent_with_fake_llm("Con")

    debate = OneOnOneDebate(
        max_loops=2,
        agents=[pro, con],
        output_type="string",
    )
    res = debate.run("Topic")
    assert isinstance(res, str)
    assert "Pro" in res
    assert "Con" in res
