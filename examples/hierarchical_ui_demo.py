"""Interactive Hierarchical Swarm UI Demo

This demo runs `HierarchicalSwarm` in interactive mode to show the
`HierarchicalSwarmDashboard` UI (Rich Live). It optionally auto-builds agents
from a prompt if LLM credentials are configured.

Run:
    python examples/hierarchical_ui_demo.py

"""

import time
import logging

from swarms.structs.hiearchical_swarm import HierarchicalSwarm

logging.basicConfig(level=logging.INFO)


def main():
    swarm = HierarchicalSwarm(
        name="HierarchicalUI",
        description="Interactive UI demo for hierarchical swarm",
        max_loops=2,
        interactive=True,
        use_parallel_execution=True,
        max_workers=6,
    )

    # Optionally auto-build a couple of agents if AutoSwarmBuilder is available
    prompt = (
        "Create 2 lightweight agents: (1) NoteTaker that summarizes notes, "
        "(2) Researcher that finds 3 data points. Return name, description, and system_prompt."
    )

    try:
        swarm.auto_build_agents_from_prompt(prompt, department_name="DemoTeam")
    except Exception:
        pass

    # Run interactive loop - will prompt when needed
    try:
        result = swarm.run(task="Gather facts about the latest AI research breakthroughs.")
        print("Run completed. Result:\n", result)
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print("Error during interactive run:", e)


if __name__ == "__main__":
    main()
