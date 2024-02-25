from swarms.agents.multion_agent import MultiOnAgent
import timeit
from swarms import Agent, ConcurrentWorkflow, Task
from swarms.utils.loguru_logger import logger

# model
model = MultiOnAgent(multion_api_key="")

# out = model.run("search for a recipe")
agent = Agent(
    agent_name="MultiOnAgent",
    description="A multi-on agent that performs browsing tasks.",
    llm=model,
    max_loops=1,
    system_prompt=None,
)

logger.info("[Agent][ID][MultiOnAgent][Initialized][Successfully")

# Task
task = Task(
    agent=agent,
    description=(
        "send an email to vyom on superhuman for a partnership with"
        " multion"
    ),
)

# Swarm
logger.info(
    f"Running concurrent workflow with task: {task.description}"
)

# Measure execution time
start_time = timeit.default_timer()

workflow = ConcurrentWorkflow(
    max_workers=1,
    autosave=True,
    print_results=True,
    return_results=True,
)

# Add task to workflow
workflow.add(task)
workflow.run()

# Calculate execution time
execution_time = timeit.default_timer() - start_time
logger.info(f"Execution time: {execution_time} seconds")
