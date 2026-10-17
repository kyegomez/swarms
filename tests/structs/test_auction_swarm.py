from unittest.mock import MagicMock
import pytest

from swarms.structs.auction_swarm import (
    AuctionSwarm,
    confidence_per_cost,
    _extract_bid,
)


def _make_agent(name: str):
    agent = MagicMock()
    agent.agent_name = name
    agent.short_memory = None
    agent.short_memory_init = MagicMock(return_value=[])
    agent.tools_list_dictionary = []
    agent.tool_choice = None
    agent.llm = MagicMock()
    agent.llm_handling = MagicMock(return_value=agent.llm)
    return agent


def test_auction_swarm_init_valid():
    a1 = _make_agent("Agent-1")
    swarm = AuctionSwarm(name="TestAuction", agents=[a1], top_k=1)
    assert swarm.name == "TestAuction"
    assert len(swarm.agents) == 1
    assert swarm.top_k == 1
    assert swarm.scoring_fn is confidence_per_cost


def test_auction_swarm_init_empty_agents_raises():
    with pytest.raises(ValueError, match="AuctionSwarm requires at least 1 agent"):
        AuctionSwarm(name="TestAuction", agents=[])


def test_auction_swarm_init_invalid_top_k():
    a1 = _make_agent("Agent-1")
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        AuctionSwarm(name="TestAuction", agents=[a1], top_k=0)


def test_auction_swarm_init_invalid_scoring():
    a1 = _make_agent("Agent-1")
    with pytest.raises(ValueError, match="Unknown scoring function"):
        AuctionSwarm(name="TestAuction", agents=[a1], scoring="nonexistent_scoring")


def test_confidence_per_cost():
    score = confidence_per_cost(confidence=0.8, estimated_cost=2.0)
    assert pytest.approx(score) == 0.4

    # cost <= 0 floor at 1e-6
    score_zero_cost = confidence_per_cost(confidence=0.8, estimated_cost=0.0)
    assert pytest.approx(score_zero_cost) == 800000.0


def test_extract_bid():
    # Tool call dict structure
    tool_call = {
        "function": {
            "name": "bid",
            "arguments": {"confidence": 0.9, "estimated_cost": 1.5},
        }
    }
    conf, cost = _extract_bid(tool_call)
    assert conf == 0.9
    assert cost == 1.5

    # List of tool calls with JSON string arguments
    tool_call_list = [
        {
            "function": {
                "name": "bid",
                "arguments": '{"confidence": 0.75, "estimated_cost": 3.0}',
            }
        }
    ]
    conf, cost = _extract_bid(tool_call_list)
    assert conf == 0.75
    assert cost == 3.0
