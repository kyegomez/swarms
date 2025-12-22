"""
Standalone Tools Example

Use the fairy tools directly without creating a full swarm.
Useful for understanding how the canvas and todo tools work.
"""

import json
from fairy_swarm import (
    SharedCanvasState,
    SharedTodoList,
    create_canvas_tool,
    create_spatial_layout_tool,
)

canvas = SharedCanvasState()
todos = SharedTodoList()

add_to_canvas = create_canvas_tool(canvas)
calculate_layout = create_spatial_layout_tool()

layout_result = calculate_layout(
    layout_type="header-body-footer",
    num_elements=3,
    canvas_width=1200,
    canvas_height=800,
)

layout = json.loads(layout_result)
for pos in layout["positions"]:
    add_to_canvas(
        element_type="wireframe-section",
        content=f"Placeholder for {pos['role']}",
        position_x=pos["x"],
        position_y=pos["y"],
        width=pos["width"],
        height=pos["height"],
    )

print(json.dumps(canvas.get_snapshot(), indent=2, default=str))
