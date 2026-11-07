from unittest.mock import Mock

import pytest

from swarms import Agent, FileChannel, GatedChannel, spec_gate
from swarms.structs.gated_channel import SPEC_QUESTIONS

SPEC = "Discount codes never stack. Apply the single best code."
MESSAGE = "Customers can stack every code they hold."


def allow(agent_name, content):
    return True, "fine"


def block(agent_name, content):
    return False, "off spec"


def broken(agent_name, content):
    raise RuntimeError("timeout")


class StubDecisionModel:
    """Answers the questions, in order, with fixed noul probabilities."""

    def __init__(self, *probabilities):
        self.probabilities = probabilities
        self.calls = []

    def run(self, state, questions):
        self.calls.append((state, questions))
        answers = zip(questions, self.probabilities)
        return {"answers": {q: {"noul": p} for q, p in answers}}


@pytest.mark.parametrize(
    "sender, gate, posted, verdict",
    [
        ("Marketing", allow, True, (True, "fine")),
        ("Marketing", block, False, (False, "off spec")),
        ("Marketing", broken, False, (False, "gate error: timeout")),
        ("Human", broken, True, None),
        ("User", broken, True, None),
        ("user", broken, True, None),
    ],
)
def test_post_reaches_the_channel_only_when_trusted_or_allowed(
    tmp_path, sender, gate, posted, verdict
):
    channel = GatedChannel(FileChannel(tmp_path / "run.md"), gate)
    channel.post(sender, MESSAGE)

    row = {"agent_name": sender, "content": MESSAGE}
    rows = []
    if verdict is not None:
        rows = [{**row, "allowed": verdict[0], "reason": verdict[1]}]
    assert channel.messages() == ([row] if posted else [])
    assert channel.verdicts == rows
    assert channel.blocked == [r for r in rows if not r["allowed"]]

    channel.clear()
    assert channel.messages() == channel.verdicts == []


def test_the_next_agent_never_reads_a_blocked_answer(tmp_path):
    checked = []

    def gate(agent_name, content):
        checked.append(content)
        return False, "off spec"

    channel = GatedChannel(FileChannel(tmp_path / "run.md"), gate)
    for name in ("Marketing", "Pricing-Engineer"):
        agent = Agent(
            agent_name=name,
            llm=Mock(),
            max_loops=1,
            print_on=False,
            persistent_memory=False,
            channel=channel,
        )
        agent.call_llm = Mock(return_value=MESSAGE)
        agent.run("Write the rule.")

    # The task is posted under Agent's user_name, which is trusted.
    assert checked == [MESSAGE, MESSAGE]
    assert channel.messages() == [
        {"agent_name": "Human", "content": "Write the rule."}
    ]
    sent = agent.call_llm.call_args.kwargs["messages"]
    assert "Write the rule." in [m["content"] for m in sent]
    assert MESSAGE not in [m["content"] for m in sent]


# Same ids, other wording, so a gate that ignores `questions` fails.
REWORDED = {
    name: {**question, "instructions": "Reworded."}
    for name, question in SPEC_QUESTIONS.items()
}


@pytest.mark.parametrize("questions", [None, REWORDED])
@pytest.mark.parametrize(
    "conflict, fact, threshold, allowed, reason",
    [
        (0.0, 0.91, 0.5, False, "fact_contradicts_spec p=0.91"),
        (0.5, 0.0, 0.5, False, "instruction_conflicts_spec p=0.50"),
        (0.0, 0.49, 0.5, True, "fact_contradicts_spec p=0.49"),
        (
            0.8,
            0.7,
            0.7,
            False,
            "instruction_conflicts_spec p=0.80, fact_contradicts_spec p=0.70",
        ),
    ],
)
def test_spec_gate_blocks_at_the_threshold_with_text_only_in_state(
    conflict, fact, threshold, allowed, reason, questions
):
    model = StubDecisionModel(conflict, fact)
    gate = spec_gate(model, SPEC, threshold, questions)

    assert gate("Marketing", MESSAGE) == (allowed, reason)
    [(state, asked)] = model.calls
    assert state == {"spec": SPEC, "message": MESSAGE}
    assert asked == (questions or SPEC_QUESTIONS)
    for question in asked.values():
        assert SPEC not in question["instructions"]
        assert MESSAGE not in question["instructions"]
