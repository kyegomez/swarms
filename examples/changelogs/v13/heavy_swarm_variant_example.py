"""HeavySwarm variant parameter.

The variant param ("default", "medium", or "heavy") replaces the old
grok-specific boolean flags.
"""

from swarms import HeavySwarm

swarm = HeavySwarm(
    variant="heavy"
)  # was: grok-specific boolean flags

result = swarm.run("Deep analysis of the AI chip market.")
print(result)
