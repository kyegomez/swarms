from dotenv import load_dotenv

from swarms import AutoAgentBuilder, SwarmRouter

load_dotenv()

TASK = (
    "Design a go-to-market strategy for an open-source developer tool "
    "moving to a paid cloud offering."
)

agents = AutoAgentBuilder(
    model_name="gpt-5.4",
    max_agents=4,
    agent_kwargs={"max_loops": 1},
).run(TASK)

print(f"Builder designed {len(agents)} agents:")
for agent in agents:
    print(f"  - {agent.agent_name}: {agent.agent_description}")

router = SwarmRouter(
    name="gtm-swarm",
    agents=agents,
    # Try "ConcurrentWorkflow", "MixtureOfAgents", "HierarchicalSwarm",
    # or "auto" to let the router decide.
    swarm_type="SequentialWorkflow",
    max_loops=1,
)

print("\n--- Result ---\n")
print(router.run(TASK))
