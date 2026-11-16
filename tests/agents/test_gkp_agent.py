from swarms.agents.gkp_agent import GKPAgent
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
