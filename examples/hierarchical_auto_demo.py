"""Hierarchical Swarm Auto-Build Demo

Demonstrates:
- Auto-building agents from a natural-language prompt using `AutoSwarmBuilder`
- Adding built agents to departments via `DepartmentManager`
- Running the hierarchical swarm with parallel order execution

Note: This demo requires LLM credentials/configuration used by the project's LiteLLM/AutoSwarmBuilder.
"""

import logging

from swarms.structs.hiearchical_swarm import HierarchicalSwarm


logging.basicConfig(level=logging.INFO)


def main():
    swarm = HierarchicalSwarm(
        name="HierarchicalAutoDemo",
        description="Demo: auto-build agents and run hierarchical swarm",
        max_loops=1,
        interactive=False,
        use_parallel_execution=True,
        max_workers=8,
    )

    prompt = (
        "Design a small 3-agent team for market analysis: "
        "(1) Researcher: gathers background and raw data, "
        "(2) DataAnalyst: analyzes numeric trends, and "
        "(3) Summarizer: writes executive summary and action items. "
        "For each agent provide name, description, system_prompt, and role."
    )

    print("Auto-building agents from prompt...")
    new_agents = swarm.auto_build_agents_from_prompt(prompt, department_name="Market")

    print("Created agents:")
    for a in new_agents:
        try:
            print(f" - {a.agent_name}: {getattr(a, 'agent_description', '')}")
        except Exception:
            print(" - <unknown agent>")

    print("Running hierarchical swarm task...")
    try:
        result = swarm.run(task="Analyze Q1 market trends and provide top 5 action items.")
        print("Swarm result:\n", result)
    except Exception as e:
        print("Error running swarm:", e)


if __name__ == "__main__":
    main()
