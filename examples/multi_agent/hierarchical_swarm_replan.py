"""
HierarchicalSwarm — incremental replan on judge rejection

When agent_as_judge=True, the judge scores every worker output after each
step.  If the collective quality is too low the judge sets verdict="REVISE"
and lists the failed agents in failed_subtasks.  The swarm then calls the
director for a ReplanAction (ADD / REASSIGN / REORDER / DROP) and re-executes
only the affected subtasks.  Successful agents are never re-run.

Run:
    python examples/multi_agent/hierarchical_swarm_replan.py
"""

from swarms import Agent
from swarms.structs.hiearchical_swarm import HierarchicalSwarm

# ---------------------------------------------------------------------------
# Worker agents
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-5"
TEMPERATURE = 1.0

research_agent = Agent(
    agent_name="ResearchAgent",
    agent_description="Finds relevant facts and data for a given topic.",
    model_name=MODEL,
    max_loops=1,
    temperature=TEMPERATURE,
)

writing_agent = Agent(
    agent_name="WritingAgent",
    agent_description="Turns research findings into a polished written report.",
    model_name=MODEL,
    max_loops=1,
    temperature=TEMPERATURE,
)

review_agent = Agent(
    agent_name="ReviewAgent",
    agent_description="Critically reviews a draft report and suggests improvements.",
    model_name=MODEL,
    max_loops=1,
    temperature=TEMPERATURE,
)

# ---------------------------------------------------------------------------
# Swarm — judge enabled, up to 2 loops so the replan result feeds back in
# ---------------------------------------------------------------------------

swarm = HierarchicalSwarm(
    name="ResearchWritingSwarm",
    description="Research, write, and review a short report on a given topic.",
    agents=[research_agent, writing_agent, review_agent],
    max_loops=2,
    agent_as_judge=True,
    judge_agent_model_name=MODEL,
    director_model_name=MODEL,
    director_temperature=TEMPERATURE,
    director_top_p=None,
    verbose=True,
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

result = swarm.run(
    task=(
        "Produce a concise 3-paragraph report on the current state of "
        "large language model research, covering: key recent advances, "
        "major open challenges, and promising future directions."
    )
)

print("\n=== Final swarm output ===\n")
print(result)
