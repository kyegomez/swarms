import json

from swarms import Agent, HierarchicalSwarm

MODEL_NAME = "claude-sonnet-5"


class UnavailableAgent(Agent):
    """Worker used to demonstrate deterministic failure recovery."""

    def run(self, *args, **kwargs):
        raise RuntimeError(
            "The primary researcher is temporarily offline."
        )


primary_researcher = UnavailableAgent(
    agent_name="Primary-Researcher",
    agent_description=(
        "The primary researcher. Assign the initial research task here first."
    ),
    model_name=MODEL_NAME,
    max_loops=1,
    persistent_memory=False,
    temperature=None,
    top_p=None,
)

backup_researcher = Agent(
    agent_name="Backup-Researcher",
    agent_description=(
        "A backup researcher that takes over tasks when another worker fails."
    ),
    system_prompt=(
        "Produce concise, factual research notes and clearly state the most "
        "important conclusions."
    ),
    model_name=MODEL_NAME,
    max_loops=1,
    persistent_memory=False,
    temperature=None,
    top_p=None,
)

swarm = HierarchicalSwarm(
    name="Recovery-Demo",
    agents=[primary_researcher, backup_researcher],
    director_settings={
        "model_name": MODEL_NAME,
        "temperature": None,
        "max_tokens": 2000,
        "top_p": None,
        "persistent_memory": False,
    },
    # Retry a failed worker once, then ask the director to reassign its task.
    max_agent_retries=1,
    max_reassignment_attempts=1,
    planning_enabled=False,
    director_feedback_on=False,
    parallel_execution=False,
    autosave=False,
    output_type="dict",
)

result = swarm.run(
    "First assign the research to Primary-Researcher. Summarize three practical "
    "benefits of multi-agent systems. If that worker is unavailable, reassign "
    "the same task to Backup-Researcher."
)

print(json.dumps(result, indent=2))
