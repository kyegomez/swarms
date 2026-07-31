"""Zena: AuctionSwarm — let agents bid for the work.

Most orchestrators have a boss decide who handles a task, which depends on
the boss's model of each worker being accurate. AuctionSwarm inverts that:
each agent self-assesses fitness for the specific task and bids.
"""

from swarms import Agent
from swarms.structs.auction_swarm import AuctionSwarm

specialists = [
    Agent(agent_name="SQL-Expert", model_name="gpt-5.4", max_loops=1),
    Agent(
        agent_name="Python-Expert", model_name="gpt-5.4", max_loops=1
    ),
    Agent(
        agent_name="Infra-Expert", model_name="gpt-5.4", max_loops=1
    ),
]

TASK = "Optimize a slow analytical query over a 40M row table."

# --- Single winner, default scoring (value per unit spend) ------------
swarm = AuctionSwarm(
    agents=specialists,
    top_k=1,
    scoring="confidence_per_cost",
)
print(swarm.run(TASK))

# --- Two winners, custom scoring that rewards confidence heavily ------
quality_first = AuctionSwarm(
    agents=specialists,
    top_k=2,
    scoring=lambda confidence, cost: confidence**2 / max(cost, 0.1),
)
print(quality_first.run(TASK))
