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


def test_a_later_task_acts_on_the_earlier_reflection(monkeypatch):
    reflection = "Always cite the source document."
    actor_prompts = []

    def fake_call_llm(self, task=None, *args, **kwargs):
        role = self.agent_name.rsplit("-", 1)[-1]
        if role == "evaluator":
            return "Final score: 9"
        if role == "reflector":
            return reflection
        actor_prompts.append(
            [
                m["content"]
                for m in self.short_memory.conversation_history
                if m["role"] == "Human"
            ][-1]
        )
        return "ok"

    monkeypatch.setattr(Agent, "call_llm", fake_call_llm)
    agent = ReflexionAgent(
        agent_name="rx", model_name="gpt-5.4", max_loops=1
    )

    agent.run(["Summarise doc A", "Summarise doc B"])

    assert (
        agent.memory.long_term_memory
    ), "the first task stored no reflection, so there is nothing to carry over"
    assert reflection in actor_prompts[1], (
        "the second task's actor never saw the stored reflection: "
        f"{actor_prompts[1]!r}"
    )
