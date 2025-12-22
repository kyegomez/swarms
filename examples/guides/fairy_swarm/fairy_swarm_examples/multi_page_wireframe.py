"""
Multi-Page Wireframe Example

A complex task demonstrating context coordination between fairies.
Creates a multi-page website wireframe with consistent elements.
"""

from fairy_swarm import FairySwarm

swarm = FairySwarm(
    name="Website Design Team",
    model_name="gpt-4o-mini",
    max_loops=3,
    enable_context_refresh=True,
    verbose=True,
)

result = swarm.run(
    "Create a multi-page website wireframe:\n"
    "1. Home page with hero section and feature highlights\n"
    "2. About page with team member cards\n"
    "3. Contact page with a form layout\n"
    "All pages must have consistent header and footer styling."
)

print(result)
