"""Hand a task off from a local agent to a remote A2A agent.

Start a2a_server_example.py first. The local agent sees the remote one
by the name and description on its agent card.
"""

from swarms import A2AClient, Agent

remote_researcher = A2AClient(
    "http://127.0.0.1:8000", api_key="demo-key"
)

coordinator = Agent(
    agent_name="Coordinator",
    system_prompt=(
        "Hand research questions off to the researcher, "
        "then summarise its answer in two sentences."
    ),
    model_name="gpt-5.4-mini",
    handoffs=[remote_researcher],
    max_loops=2,
)

if __name__ == "__main__":
    print(
        coordinator.run("Find out what the A2A protocol is used for.")
    )
