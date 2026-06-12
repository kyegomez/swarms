"""RoundRobinSwarm true rotation (v12.0.2).

Deterministic rotation replaces the previous shuffle-based ordering:
Handler-0 -> Handler-1 -> Handler-2. Each agent receives previous/next
speaker context (turn awareness), and logs are gated behind verbose.
"""

from swarms import Agent, RoundRobinSwarm

agents = [
    Agent(
        agent_name=f"Handler-{i}", model_name="gpt-4.1", max_loops=1
    )
    for i in range(3)
]

# Fixed order: Handler-0 -> Handler-1 -> Handler-2; logs only when verbose=True
rr = RoundRobinSwarm(agents=agents, max_loops=1, verbose=True)

# Each agent knows who spoke before it and who speaks next
result = rr.run("Review this proposal.")
print(result)
