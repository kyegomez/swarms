import threading

import pytest

from swarms.structs import checked_swarm as cs
from swarms.structs.checked_swarm import CheckedSwarm


class FakeAgent:
    """Offline template whose every fresh copy plays the next script."""

    def __init__(self, agent_name, *scripts):
        self.agent_name = agent_name
        self.scripts = list(scripts)
        self.hired, self.prompts, self.replies = [], [], []

    def run(self, task):
        self.prompts.append(task)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply() if callable(reply) else reply


@pytest.fixture(autouse=True)
def scripted_copies(monkeypatch):
    def fresh(template):
        agent = FakeAgent(template.agent_name)
        agent.replies = template.scripts.pop(0)
        template.hired.append(agent)
        return agent

    monkeypatch.setattr(cs, "fresh_agent", fresh)


def is_42(answer):
    return int(answer) == 42, "the product is wrong"


@pytest.mark.parametrize(
    "scripts, events, output, last_prompt_has",
    [
        ([["42\nSTATUS: DONE"]], ["verified"], "42", "Task:"),
        (
            [["41\nSTATUS: DONE"], ["42\nSTATUS: DONE"]],
            ["false_claim", "verified"],
            "42",
            "claimed done but the check failed: the product is wrong",
        ),
        (
            [[RuntimeError("model down"), "42\n**STATUS: DONE**"]],
            ["retry", "verified"],
            "42",
            "failed: The agent raised RuntimeError: model down",
        ),
        (
            [["forty-two", "42\nSTATUS: DONE"]],
            ["retry", "verified"],
            "42",
            "failed: ValueError: invalid literal",
        ),
        (
            [["41", "41\nSTATUS: STUCK: no data"], ["43", "44"]],
            ["retry", "stuck", "retry", "stuck", "failed"],
            "Unverified: the product is wrong",
            "could not pass the check: the product is wrong",
        ),
    ],
    ids=["honest", "liar", "raises", "check_raises", "stuck"],
)
def test_an_answer_counts_only_when_its_check_passes(
    scripts, events, output, last_prompt_has
):
    template = FakeAgent("Solver", *scripts)
    swarm = CheckedSwarm(
        agents=[template],
        checks=[is_42],
        max_attempts=2,
        output_type="final",
    )

    assert swarm.run("What is six times seven?") == output
    assert [row["event"] for row in swarm.last_run] == events
    assert all(
        (row["reason"] == "") == (row["event"] == "verified")
        for row in swarm.last_run
    )
    assert last_prompt_has in template.hired[-1].prompts[-1]
    for replacement in template.hired[1:]:
        assert not any("41" in p for p in replacement.prompts)


def test_a_lesson_reaches_the_other_parts():
    replaced = threading.Event()
    product = FakeAgent(
        "Multiplier",
        ["41\nSTATUS: DONE"],
        [lambda: replaced.set() or "42\nSTATUS: DONE"],
    )
    summary = FakeAgent("Writer", ["draft", "final\nSTATUS: DONE"])

    def is_final(answer):
        replaced.wait(timeout=5)
        return answer == "final", "not final"

    CheckedSwarm(
        agents=[product, summary],
        checks=[is_42, is_final],
        parts=["the product", "the summary"],
    ).run("Multiply six by seven and summarise.")

    lesson = 'An earlier agent on "the product" claimed done'
    assert lesson in summary.hired[0].prompts[1]
