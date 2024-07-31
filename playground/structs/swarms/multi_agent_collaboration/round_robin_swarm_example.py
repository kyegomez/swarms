from swarms.structs.round_robin import RoundRobinSwarm
from swarms import Agent, OpenAIChat


# Initialize the LLM
llm = OpenAIChat()

# Define sales agents
sales_agent1 = Agent(
    agent_name="Sales Agent 1 - Automation Specialist",
    system_prompt="You're Sales Agent 1, your purpose is to generate sales for a company by focusing on the benefits of automating accounting processes!",
    agent_description="Generate sales by focusing on the benefits of automation!",
    llm=llm,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    context_length=1000,
)

sales_agent2 = Agent(
    agent_name="Sales Agent 2 - Cost Saving Specialist",
    system_prompt="You're Sales Agent 2, your purpose is to generate sales for a company by emphasizing the cost savings of using swarms of agents!",
    agent_description="Generate sales by emphasizing cost savings!",
    llm=llm,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    context_length=1000,
)

sales_agent3 = Agent(
    agent_name="Sales Agent 3 - Efficiency Specialist",
    system_prompt="You're Sales Agent 3, your purpose is to generate sales for a company by highlighting the efficiency and accuracy of our swarms of agents in accounting processes!",
    agent_description="Generate sales by highlighting efficiency and accuracy!",
    llm=llm,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    context_length=1000,
)

# Initialize the swarm with sales agents
sales_swarm = RoundRobinSwarm(
    agents=[sales_agent1, sales_agent2, sales_agent3], verbose=True
)

# Define a sales task
task = "Generate a sales email for an accountant firm executive to sell swarms of agents to automate their accounting processes."

# Distribute sales tasks to different agents
for _ in range(5):  # Repeat the task 5 times
    results = sales_swarm.run(task)
    print("Sales generated:", results)
