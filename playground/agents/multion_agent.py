from swarms.agents.multion_agent import MultiOnAgent
from swarms.structs.agent import Agent
from swarms.structs.concurrent_workflow import ConcurrentWorkflow
from swarms.structs.task import Task

# model
model = MultiOnAgent(
    multion_api_key=""
)

# out = model.run("search for a recipe")
agent = Agent(
    agent_name="MultiOnAgent",
    description="A multi-on agent that performs browsing tasks.",
    llm=model,
    max_loops=1,
    system_prompt=None,
)


# Task
task = Task(
    agent=agent,
    description=(
        "send an email to vyom on superhuman for a partnership with"
        " multion"
    ),
)


# Swarm
workflow = ConcurrentWorkflow(
    max_workers=21,
    autosave=True,
    print_results=True,
    return_results=True,
)


# Add task to workflow
workflow.add(task)
workflow.run()
