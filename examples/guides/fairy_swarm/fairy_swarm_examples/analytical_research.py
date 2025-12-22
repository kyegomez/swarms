"""
Analytical Research Example

Use the Analytical Fairy's strengths for research-heavy tasks.
The orchestrator will delegate analysis work appropriately.
"""

from fairy_swarm import FairySwarm

swarm = FairySwarm(
    name="Research Team",
    model_name="gpt-4o-mini",
    max_loops=2,
    verbose=True,
)

result = swarm.run(
    "Research and design an information dashboard for tracking:\n"
    "1. Key performance metrics with charts\n"
    "2. Data tables with sorting capabilities\n"
    "3. Filter controls and search functionality\n"
    "Include recommendations for data visualization best practices."
)

print(result)
