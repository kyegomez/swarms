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
