"""
Dashboard dirty-panel example.

Shows that HierarchicalSwarmDashboard only rebuilds the panel that changed
instead of the full layout on every update call.
"""

import time

from swarms.structs.hiearchical_swarm import HierarchicalSwarmDashboard

if __name__ == "__main__":
    dashboard = HierarchicalSwarmDashboard(swarm_name="Demo-Swarm")
    dashboard.start(max_loops=3)

    dashboard.update_swarm_info(
        name="Demo-Swarm",
        description="Dirty-panel refresh demo",
        max_loops=3,
        director_name="Director",
        director_model_name="gpt-4.1",
    )

    dashboard.update_director_plan(
        "Step 1: research. Step 2: analyse. Step 3: report."
    )
    dashboard.update_director_orders(
        [
            {"agent_name": "Researcher", "task": "Gather data on AI chips"},
            {"agent_name": "Analyst", "task": "Summarise findings"},
        ]
    )

    agents = ["Researcher", "Analyst", "Writer"]
    for agent in agents:
        dashboard.update_agent_status(
            agent, "RUNNING", task=f"{agent} initial task"
        )
        time.sleep(0.4)

    # Simulate loop progression — only operations_status panel refreshes
    for loop in range(1, 4):
        dashboard.update_loop(loop)
        time.sleep(0.3)

    # Complete agents one by one — only agents panel refreshes
    for agent in agents:
        dashboard.update_agent_status(
            agent,
            "COMPLETED",
            task=f"{agent} task",
            output=f"{agent} output text",
        )
        time.sleep(0.4)

    dashboard.update_director_status("COMPLETED")
    time.sleep(1)
    dashboard.stop()
    print("Done.")
