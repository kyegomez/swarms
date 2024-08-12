from swarms import Agent, llama3Hosted
from swarms.structs.swarm_load_balancer import AgentLoadBalancer

# Initialize the language model agent (e.g., GPT-3)
llm = llama3Hosted()

# Initialize agents for individual tasks
agent1 = Agent(
    agent_name="Blog generator",
    system_prompt="Generate a blog post like stephen king",
    llm=llm,
    max_loops=1,
    dashboard=False,
)
agent2 = Agent(
    agent_name="Summarizer",
    system_prompt="Sumamrize the blog post",
    llm=llm,
    max_loops=1,
    dashboard=False,
)

# Create the Sequential workflow
workflow = AgentLoadBalancer(
    agents=[agent1, agent2],
    max_loops=1,
)

# Run the workflow
workflow.run(
    "Generate a blog post on how swarms of agents can help businesses grow."
)
