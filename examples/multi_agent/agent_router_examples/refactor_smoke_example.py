"""Minimal example for `AgentRouter`.

Adds three agents to an embedding-based router and updates one agent's
history (the path that the refactor touched).
"""

from swarms import Agent
from swarms.structs.agent_router import AgentRouter


def main() -> None:
    researcher = Agent(
        agent_name="Researcher",
        agent_description="Researches topics in depth.",
        model_name="gpt-5.4",
        max_loops=1,
        print_on=False,
    )
    coder = Agent(
        agent_name="Coder",
        agent_description="Writes and reviews code.",
        model_name="gpt-5.4",
        max_loops=1,
        print_on=False,
    )
    writer = Agent(
        agent_name="Writer",
        agent_description="Drafts clean prose.",
        model_name="gpt-5.4",
        max_loops=1,
        print_on=False,
    )

    router = AgentRouter(
        embedding_model="text-embedding-3-small",
        n_agents=1,
        agents=[researcher, coder, writer],
    )

    # Exercise the path that previously did two linear scans.
    router.update_agent_history("Coder")

    # Find the best agent for a task.
    match = router.find_best_agent("Help me write a Python function.")
    print("best match:", getattr(match, "agent_name", match))


if __name__ == "__main__":
    main()
