from swarms.agents.llm_manager import DEFAULT_MODEL_NAME
from swarms.structs.agent import Agent
from swarms.structs.mixture_of_agents import MixtureOfAgents


def test_mixture_of_agents_default_aggregator_model_name():
    """MixtureOfAgents.aggregator_model_name must default to DEFAULT_MODEL_NAME."""
    w1 = Agent(agent_name="W1", model_name="gpt-5.4")
    moa = MixtureOfAgents(agents=[w1], layers=1)
    assert moa.aggregator_model_name == DEFAULT_MODEL_NAME
    assert moa.aggregator_agent.model_name == DEFAULT_MODEL_NAME


def test_mixture_of_agents_custom_aggregator_model_name():
    """Custom aggregator_model_name should still be honored."""
    w1 = Agent(agent_name="W1", model_name="gpt-5.4")
    moa = MixtureOfAgents(
        agents=[w1], layers=1, aggregator_model_name="custom-model"
    )
    assert moa.aggregator_model_name == "custom-model"
    assert moa.aggregator_agent.model_name == "custom-model"
