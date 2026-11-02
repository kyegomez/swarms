import json

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


def _bidder(name, confidence, output_type="str-all-except-first"):
    agent = Agent(
        agent_name=name, print_on=False, output_type=output_type
    )

    def call_llm(task=None, *args, **kwargs):
        if agent.tools_list_dictionary:
            call = {
                "name": "bid",
                "arguments": json.dumps({"confidence": confidence}),
            }
            return [{"function": call}]
        return f"answer from {name}"

    agent.call_llm = call_llm
    return agent


@pytest.mark.parametrize("kind", ["str-all-except-first", "dict"])
def test_best_bidder_wins_every_run_whatever_its_output_type(kind):
    swarm = AuctionSwarm(
        agents=[_bidder("Best", 0.9, kind), _bidder("Other", 0.5)],
        output_type="final",
        print_on=False,
    )

    for task in ("Translate clause 1.", "Translate clause 2."):
        assert swarm.run(task) == "answer from Best"
