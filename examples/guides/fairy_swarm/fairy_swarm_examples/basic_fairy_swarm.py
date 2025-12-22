"""
Basic FairySwarm Example

A simple example showing how to create a FairySwarm and run a design task.
The swarm coordinates multiple fairy agents to work on a shared canvas.

Inspired by tldraw's fairies feature for multi-agent coordination.
"""

from fairy_swarm import FairySwarm

swarm = FairySwarm(
    name="Design Team",
    description="A collaborative team of fairies for UI design",
    model_name="gpt-4o-mini",
    max_loops=2,
    verbose=True,
)

result = swarm.run(
    "Create a wireframe for a simple landing page with:\n"
    "1. A header with logo and navigation menu\n"
    "2. A hero section with headline and CTA button\n"
    "3. A footer with copyright and social links"
)

print(result)
