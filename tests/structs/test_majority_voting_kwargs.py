import inspect
from typing import Optional, Dict, Any
from swarms.structs.agent import Agent
from swarms.structs.majority_voting import MajorityVoting


def test_additional_consensus_agent_kwargs_default_is_none():
    """Default argument in MajorityVoting.__init__ must be None, not a shared mutable dict."""
    sig = inspect.signature(MajorityVoting.__init__)
    param = sig.parameters["additional_consensus_agent_kwargs"]
    assert param.default is None, f"Expected None, got {param.default}"


def test_additional_consensus_agent_kwargs_none_accepted():
    """Passing additional_consensus_agent_kwargs=None explicitly must not raise TypeError."""
    dummy_agent = Agent(agent_name="Agent-1", max_loops=1)
    mv = MajorityVoting(
        name="test-mv",
        agents=[dummy_agent],
        additional_consensus_agent_kwargs=None,
    )
    assert mv.consensus_agent is not None


def test_additional_consensus_agent_kwargs_passed_to_consensus_agent():
    """Provided kwargs must be forwarded to the consensus agent."""
    dummy_agent = Agent(agent_name="Agent-1", max_loops=1)
    mv = MajorityVoting(
        name="test-mv",
        agents=[dummy_agent],
        additional_consensus_agent_kwargs={"context_length": 8192},
    )
    assert mv.consensus_agent.context_length == 8192
