"""A coordinator agent opens threads; a swarm then works in each one."""

from concurrent.futures import ThreadPoolExecutor

from swarms import (
    Agent,
    ConcurrentWorkflow,
    MessageBoard,
    SequentialWorkflow,
)

board = MessageBoard(
    name="projects",
    description="One thread per project, worked by a team's swarm",
)

coordinator = Agent(
    agent_name="Coordinator",
    system_prompt=(
        "You open one message board thread per project with "
        "create_thread. Label each one team=research or team=marketing."
    ),
    model_name="gpt-5.4",
    max_loops=2,
    tools=board.agent_tools,
)
coordinator.run(
    "Open two projects: a Q3 report on the EV battery market for the "
    "research team, and taglines for a budget electric scooter launch for "
    "the marketing team."
)


def agent(name: str, system_prompt: str) -> Agent:
    """Build a one-loop agent; each swarm gets its own agents."""
    return Agent(
        agent_name=name,
        system_prompt=system_prompt,
        model_name="gpt-5.4",
        max_loops=1,
    )


def research_swarm(thread_id: str) -> SequentialWorkflow:
    """A pipeline that posts each step to the thread."""
    return SequentialWorkflow(
        agents=[
            agent("Researcher", "Gather the key facts as bullets."),
            agent("Writer", "Turn the facts into a short report."),
            agent("Editor", "Tighten the report."),
        ],
        max_loops=1,
        channel=board.channel(thread_id),
    )


def marketing_swarm(thread_id: str) -> ConcurrentWorkflow:
    """A fan-out whose copywriters each post their own ideas."""
    return ConcurrentWorkflow(
        agents=[
            agent("Bold-Copywriter", "Write three bold taglines."),
            agent(
                "Playful-Copywriter", "Write three playful taglines."
            ),
        ],
        channel=board.channel(thread_id),
    )


builders = {"research": research_swarm, "marketing": marketing_swarm}

# Each thread gets the swarm its team label asks for, all running at once.
with ThreadPoolExecutor() as pool:
    for thread in board.threads():
        build = builders.get(thread["labels"].get("team"))
        if build is not None:
            pool.submit(
                build(thread["id"]).run,
                f"Do the project in this thread: {thread['title']}",
            )

for thread in board.threads():
    print(f"=== {thread['title']} ({thread['messages']} messages)")
    print(board.thread(thread["id"]).return_history_as_string())
    print()
