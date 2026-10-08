"""Zena: HierarchicalSwarm worker recovery, planning, and a judge.

A failing worker is retried up to max_agent_retries. If it stays
unavailable, the director may reassign its task up to
max_reassignment_attempts times rather than dropping the work.

Director behavior is configurable without subclassing.
"""

from swarms import Agent, HierarchicalSwarm

director = Agent(
    agent_name="Director", model_name="gpt-5.4", max_loops=1
)
workers = [
    Agent(
        agent_name="DataWorker",
        model_name="gpt-5.4-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="WritingWorker",
        model_name="gpt-5.4-mini",
        max_loops=1,
    ),
]

swarm = HierarchicalSwarm(
    director=director,
    agents=workers,
    max_loops=1,
    planning_enabled=True,  # plan before delegating
    agent_as_judge=True,  # score worker output
    max_agent_retries=2,  # retry a failing worker
    max_reassignment_attempts=1,  # then reassign the task
    parallel_execution=True,
    max_workers=8,
    # Director overrides, no subclassing required
    director_model_name="claude-sonnet-4-6",
    director_temperature=0.2,
    director_settings={
        "max_tokens": 16000,
        "reasoning_effort": "high",
    },
)

print(
    swarm.run("Produce a competitive analysis of the AI chip market.")
)
