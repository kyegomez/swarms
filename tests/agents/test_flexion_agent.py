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
