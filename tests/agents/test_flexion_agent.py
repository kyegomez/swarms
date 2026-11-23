from swarms.agents.flexion_agent import ReflexionAgent
from swarms.structs.agent import Agent


def test_each_iteration_records_only_the_latest_replies(monkeypatch):
    calls = {"actor": 0, "evaluator": 0, "reflector": 0}

    def fake_call_llm(self, task=None, *args, **kwargs):
        role = self.agent_name.rsplit("-", 1)[-1]
        calls[role] += 1
        n = calls[role]
        if role == "actor":
            return f"answer {n}"
        if role == "evaluator":
            return f"Final score: {3 + 5 * (n - 1)}"
        return f"reflection {n}"

    monkeypatch.setattr(Agent, "call_llm", fake_call_llm)
    agent = ReflexionAgent(
        agent_name="rx", model_name="gpt-5.4", max_loops=2
    )

    iterations = agent.run(
        ["What is 2+2?"], include_intermediates=True
    )[0]

    assert [
        (step["response"], step["evaluation"], step["reflection"])
        for step in iterations
    ] == [
        ("answer 1", "Final score: 3", "reflection 1"),
        ("answer 2", "Final score: 8", "reflection 2"),
    ]


def test_each_call_sees_only_its_own_prompt(monkeypatch):
    requests = []

    def fake_call_llm(self, task=None, *args, **kwargs):
        prompts = [
            m["content"]
            for m in self.short_memory.conversation_history
            if m["role"] == "Human"
        ]
        requests.append((self.agent_name, prompts))
        if self.agent_name.endswith("evaluator"):
            return "Final score: 5"
        return "ok"

    monkeypatch.setattr(Agent, "call_llm", fake_call_llm)
    agent = ReflexionAgent(
        agent_name="rx", model_name="gpt-5.4", max_loops=2
    )

    agent.run(["Summarise doc A", "Write a haiku"])

    assert len(requests) == 12
    for name, prompts in requests:
        assert len(prompts) == 1, f"{name} also saw {prompts[:-1]}"


def test_usage_sums_every_agent_call_across_runs(monkeypatch):
    calls = []

    def fake_call_llm(self, task=None, *args, **kwargs):
        calls.append(self.agent_name)
        self._add_usage(
            {
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
            }
        )
        if self.agent_name.endswith("evaluator"):
            return "Final score: 3"
        return "text"

    monkeypatch.setattr(Agent, "call_llm", fake_call_llm)
    agent = ReflexionAgent(
        agent_name="rx", model_name="gpt-5.4", max_loops=2
    )
    assert agent.usage == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }

    agent.run(["What is 2+2?"])
    first = len(calls)
    assert {name.rsplit("-", 1)[-1] for name in calls} == {
        "actor",
        "evaluator",
        "reflector",
    }
    assert agent.usage["input_tokens"] == 10 * first

    agent.run(["What is 3+3?"])

    assert len(calls) > first
    assert agent.usage == {
        "input_tokens": 10 * len(calls),
        "output_tokens": 2 * len(calls),
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 12 * len(calls),
    }
