from dotenv import load_dotenv

from swarms import AutoAgentBuilder, ConcurrentWorkflow

load_dotenv()

TASK = (
    "Assess the risks of deploying autonomous AI agents in a hospital's "
    "clinical workflow. Cover clinical safety, regulatory exposure, and "
    "patient privacy."
)

# One agent per risk domain — they do not depend on each other, so there is no
# reason to run them in sequence.
agents = AutoAgentBuilder(
    model_name="gpt-5.4",
    num_agents=3,
    agent_kwargs={"max_loops": 1},
).run(TASK)

print(f"Running {len(agents)} agents concurrently:")
for agent in agents:
    print(f"  - {agent.agent_name}")

results = ConcurrentWorkflow(agents=agents).run(TASK)

print("\n--- Results ---\n")
print(results)
