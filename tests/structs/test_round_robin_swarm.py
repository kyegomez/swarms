import os
import pytest
from swarms.structs.round_robin import RoundRobinSwarm
from swarms.structs.agent import Agent


@pytest.fixture
def round_robin_swarm():
    agents = [Agent(name=f"Agent{i}") for i in range(3)]
    return RoundRobinSwarm(agents=agents, verbose=True, max_loops=2)



# Live-LLM tests: they call agent.run() against a real model and can only
# pass with an API key. Without one, Agent.run now honestly raises
# AgentLLMError after retry exhaustion, so these tests are skipped instead
# of failing (same convention as tests/telemetry/test_telemetry.py).
_LLM_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
)
_HAS_LLM_KEY = any(os.getenv(k) for k in _LLM_KEYS)
requires_llm = pytest.mark.skipif(
    not _HAS_LLM_KEY,
    reason="no LLM API key set (OPENAI_API_KEY etc.) - live-LLM test skipped",
)

def test_init(round_robin_swarm):
    assert isinstance(round_robin_swarm, RoundRobinSwarm)
    assert round_robin_swarm.verbose is True
    assert round_robin_swarm.max_loops == 2
    assert len(round_robin_swarm.agents) == 3


@requires_llm
def test_run(round_robin_swarm):
    task = "test_task"
    result = round_robin_swarm.run(task)
    assert result == task
    assert round_robin_swarm.index == 0
