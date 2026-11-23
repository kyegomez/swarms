from swarms.agents.gkp_agent import GKPAgent, KnowledgeGenerator
from swarms.structs.agent import Agent


def test_each_path_and_each_query_gets_its_own_answer(monkeypatch):
    paths = []

    def fake_call_llm(self, task=None, *args, **kwargs):
        prompt = [
            message["content"]
            for message in self.short_memory.conversation_history
            if message["role"] == "Human"
        ][-1]
        city = "Canberra" if "Australia" in prompt else "Paris"
        if self.agent_name.endswith("knowledge-generator"):
            return f"Knowledge 1: {city} one.\n\nKnowledge 2: {city} two."
        if self.agent_name.endswith("coordinator"):
            return f"Analysis: paths agree.\nFinal Answer: {city}"
        paths.append(city)
        return f"Explanation: path {len(paths)}.\nConfidence: high\nAnswer: {city}"

    monkeypatch.setattr(Agent, "call_llm", fake_call_llm)
    gkp = GKPAgent(
        agent_name="gkp", model_name="gpt-5.4", num_knowledge_items=2
    )

    detail = gkp.process("What is the capital of Australia?")

    assert [
        (result["explanation"], result["answer"])
        for result in detail["reasoning_results"]
    ] == [("path 1.", "Canberra"), ("path 2.", "Canberra")]
    assert gkp.run("What is the capital of France?") == "Paris"


def test_each_call_sees_only_its_own_prompt(monkeypatch):
    requests = []

    def fake_call_llm(self, task=None, *args, **kwargs):
        history = self.short_memory.conversation_history
        prompts = [
            m["content"] for m in history if m["role"] == "Human"
        ]
        requests.append((self.agent_name, prompts))
        if self.agent_name.endswith("knowledge-generator"):
            city = (
                "Canberra" if "Australia" in prompts[-1] else "Paris"
            )
            return f"Knowledge 1: {city} one.\n\nKnowledge 2: {city} two."
        if self.agent_name.endswith("coordinator"):
            return "Analysis: ok.\nFinal Answer: done"
        return "Explanation: e.\nConfidence: high\nAnswer: a"

    monkeypatch.setattr(Agent, "call_llm", fake_call_llm)
    gkp = GKPAgent(
        agent_name="gkp", model_name="gpt-5.4", num_knowledge_items=2
    )

    gkp.run("What is the capital of Australia?")
    gkp.run("What is the capital of France?")

    assert len(requests) == 8
    for name, prompts in requests:
        assert len(prompts) == 1, f"{name} also saw {prompts[:-1]}"


def test_knowledge_generator_passes_its_description_to_the_agent():
    generator = KnowledgeGenerator(description="Finds dates.")

    assert generator.agent.agent_description == "Finds dates."


def test_usage_sums_generation_reasoning_and_coordination_across_runs(
    monkeypatch,
):
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
        if self.agent_name.endswith("knowledge-generator"):
            return "Knowledge 1: one.\n\nKnowledge 2: two."
        if self.agent_name.endswith("coordinator"):
            return "Analysis: agree.\nFinal Answer: Paris"
        return "Explanation: e.\nConfidence: high\nAnswer: Paris"

    monkeypatch.setattr(Agent, "call_llm", fake_call_llm)
    gkp = GKPAgent(
        agent_name="gkp", model_name="gpt-5.4", num_knowledge_items=2
    )
    assert gkp.usage == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }

    gkp.run("What is the capital of France?")
    first = len(calls)
    assert gkp.knowledge_generator.usage["input_tokens"] > 0
    assert gkp.reasoner.usage["input_tokens"] > 0
    assert gkp.coordinator.usage["input_tokens"] > 0
    gkp.run("What is the capital of France?")

    assert len(calls) == 2 * first
    assert gkp.usage == {
        "input_tokens": 10 * len(calls),
        "output_tokens": 2 * len(calls),
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 12 * len(calls),
    }
