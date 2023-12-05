from swarms.swarms.multi_agent_debate import (
    MultiAgentDebate,
    select_speaker,
)
from swarms.workers.worker import Worker
from swarms.models import OpenAIChat

llm = OpenAIChat()

worker1 = Worker(
    llm=llm,
    ai_name="Bumble Bee",
    ai_role="Worker in a swarm",
    external_tools=None,
    human_in_the_loop=False,
    temperature=0.5,
)
worker2 = Worker(
    llm=llm,
    ai_name="Optimus Prime",
    ai_role="Worker in a swarm",
    external_tools=None,
    human_in_the_loop=False,
    temperature=0.5,
)
worker3 = Worker(
    llm=llm,
    ai_name="Megatron",
    ai_role="Worker in a swarm",
    external_tools=None,
    human_in_the_loop=False,
    temperature=0.5,
)

# init agents
agents = [worker1, worker2, worker3]

# Initialize multi-agent debate with the selection function
debate = MultiAgentDebate(agents, select_speaker)

# Run task
task = (
    "What were the winning boston marathon times for the past 5 years"
    " (ending in 2022)? Generate a table of the year, name, country"
    " of origin, and times."
)
results = debate.run(task, max_iters=4)

# Print results
for result in results:
    print(f"Agent {result['agent']} responded: {result['response']}")
