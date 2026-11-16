import pytest

from swarms.structs import auction_swarm
from swarms.structs.agent import Agent
from swarms.structs.auction_swarm import AuctionSwarm


def test_run_raises_when_every_winner_fails(monkeypatch):
    agents = [
        Agent(agent_name=name, model_name="gpt-5.4", print_on=False)
        for name in ("A", "B")
    ]
    swarm = AuctionSwarm(
        agents=agents, top_k=2, output_type="final", print_on=False
    )
    monkeypatch.setattr(
        swarm,
        "_run_auction",
        lambda task: [(agent, 0.9, 1.0, 0.9) for agent in agents],
    )
    monkeypatch.setattr(
        auction_swarm,
        "run_agents_concurrently",
        lambda agents, **kwargs: {
            agent.agent_name: RuntimeError("provider down")
            for agent in agents
        },
    )

    with pytest.raises(
        RuntimeError, match="every winning agent"
    ) as info:
        swarm.run("Summarise the report")

    assert str(info.value.__cause__) == "provider down"
