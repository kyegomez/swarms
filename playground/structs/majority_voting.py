from swarms import Agent, MajorityVoting, OpenAIChat

# Initialize the llm
llm = OpenAIChat()

# Initialize the agents
agent1 = Agent(agent_name="worker-1", llm=llm, max_loops=1)
agent2 = Agent(agent_name="worker-2", llm=llm, max_loops=1)
agent3 = Agent(agent_name="worker3", llm=llm, max_loops=1)


# Initialize the majority voting
mv = MajorityVoting(
    agents=[agent1, agent2, agent3],
    concurrent=True,
    multithreaded=True,
)


# Start the majority voting
mv.run("What is the capital of France?")
