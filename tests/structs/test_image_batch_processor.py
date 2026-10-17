import threading
from unittest.mock import patch

from swarms.structs.agent import Agent
from swarms.structs.image_batch_processor import (
    ImageAgentBatchProcessor,
)


def test_results_follow_input_order_and_name_their_agent(tmp_path):
    image = tmp_path / "chart.png"
    image.touch()
    fast_done = threading.Event()

    def fake_call_llm(self, task=None, *args, **kwargs):
        if self.agent_name == "Slow":
            fast_done.wait(timeout=5)
            return "slow answer"
        fast_done.set()
        return "fast answer"

    agents = [
        Agent(agent_name=n, print_on=False) for n in ("Slow", "Fast")
    ]
    processor = ImageAgentBatchProcessor(agents=agents, max_workers=2)
    with patch.object(Agent, "call_llm", fake_call_llm):
        results = processor.run(str(image), "Describe")

    answers = [r["results"]["Describe"] for r in results]
    assert fast_done.is_set()
    assert "slow answer" in answers[0] and "fast answer" in answers[1]
    assert [r["agent"] for r in results] == ["Slow", "Fast"]
