"""
Selected Fairies Example

Run a task with a subset of fairies where one is elected as orchestrator.
This mimics tldraw's behavior when you select multiple fairies and prompt the group.

The first fairy in the list becomes the orchestrator for the task.
"""

from fairy_swarm import FairySwarm

swarm = FairySwarm(
    name="Creative Duo",
    model_name="gpt-4o-mini",
    max_loops=2,
    verbose=True,
)

result = swarm.run_with_selected_fairies(
    task="Design a colorful banner for a summer music festival with bold typography and vibrant imagery",
    fairy_names=["Creative-Fairy", "Harmonizer-Fairy"],
)

print(result)
