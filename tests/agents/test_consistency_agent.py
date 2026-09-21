"""Sample independence for ``SelfConsistencyAgent``."""

import threading
import time

from swarms.agents.consistency_agent import SelfConsistencyAgent
from swarms.structs.agent import Agent


def _run_samples(num_samples):
    """Run the agent once, recording each sample's agent object and prompt."""
    samples = []
    lock = threading.Lock()
    counter = iter(range(1, 1000))

    def fake_call_llm(self, task=None, *args, **kwargs):
        index = next(counter)
        time.sleep(0.02 * index)
        with lock:
            samples.append(
                {
                    "agent_id": id(self),
                    "prompt": self.short_memory.return_history_as_string(),
                }
            )
        return f"ANSWER_{index}"

    original = Agent.call_llm
    Agent.call_llm = fake_call_llm
    try:
        agent = SelfConsistencyAgent(
            name="Sampler",
            model_name="gpt-5.4",
            num_samples=num_samples,
        )
        agent.run("What is 2 + 2?")
    finally:
        Agent.call_llm = original

    return samples[:num_samples]


def test_each_sample_gets_its_own_agent():
    """One shared agent let every sample write into the same short_memory."""
    samples = _run_samples(3)

    agent_ids = {sample["agent_id"] for sample in samples}
    assert len(agent_ids) == 3


def test_no_sample_sees_an_earlier_samples_answer():
    """Samples were sequentially conditioned, so the majority vote was not over independent draws."""
    samples = _run_samples(3)

    for sample in samples:
        for index in range(1, 4):
            assert f"ANSWER_{index}" not in sample["prompt"]
