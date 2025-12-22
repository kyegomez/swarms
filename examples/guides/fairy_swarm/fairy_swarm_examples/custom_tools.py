"""
Custom Tools Example

Add custom tools to fairies for specialized functionality.
This example adds a color palette generator tool.
"""

import json
from fairy_swarm import FairySwarm


def generate_color_palette(theme: str, num_colors: int = 5) -> str:
    """
    Generate a color palette for a given theme.

    Args:
        theme: The theme for the palette (ocean, sunset, forest)
        num_colors: Number of colors to generate

    Returns:
        JSON string with the color palette
    """
    palettes = {
        "ocean": [
            "#0077B6",
            "#00B4D8",
            "#90E0EF",
            "#CAF0F8",
            "#03045E",
        ],
        "sunset": [
            "#FF6B6B",
            "#FFA06B",
            "#FFD93D",
            "#FF8E53",
            "#C44536",
        ],
        "forest": [
            "#2D5A27",
            "#5B8C5A",
            "#8BC34A",
            "#C8E6C9",
            "#1B5E20",
        ],
    }
    colors = palettes.get(theme.lower(), palettes["ocean"])[
        :num_colors
    ]
    return json.dumps({"theme": theme, "colors": colors})


swarm = FairySwarm(
    name="Color-Aware Design Team",
    model_name="gpt-4o-mini",
    max_loops=2,
    verbose=True,
    additional_tools=[generate_color_palette],
)

result = swarm.run(
    "Design a landing page for an ocean-themed travel agency. "
    "Use the color palette tool to get appropriate colors for the design."
)

print(result)
