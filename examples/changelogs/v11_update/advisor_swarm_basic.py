from swarms import AdvisorSwarm

# Pair a cheap executor with a powerful advisor
swarm = AdvisorSwarm(
    executor_model_name="claude-sonnet-4-6",
    advisor_model_name="claude-opus-4-6",
    max_advisor_uses=3,  # budget: how many times advisor is consulted
    max_loops=5,  # executor turns
)

result = swarm.run(
    "Design and implement a rate-limiting middleware in Python"
)
print(result)
