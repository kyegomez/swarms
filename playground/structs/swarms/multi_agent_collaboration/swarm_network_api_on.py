from swarms import Agent, OpenAIChat, SwarmNetwork

# Create an instance of Agent
agent1 = Agent(
    agent_name="Covid-19-Chat",  # Name of the agent
    agent_description="This agent provides information about COVID-19 symptoms.",  # Description of the agent
    llm=OpenAIChat(),  # Language model used by the agent
    max_loops="auto",  # Maximum number of loops the agent can run
    autosave=True,  # Whether to automatically save the agent's state
    verbose=True,  # Whether to print verbose output
    stopping_condition="finish",  # Condition for stopping the agent
)

agents = [agent1]  # List of agents (add more agents as needed)

swarm_name = "HealthSwarm"  # Name of the swarm
swarm_description = "A swarm of agents providing health-related information."  # Description of the swarm

# Create an instance of SwarmNetwork with API enabled
agent_api = SwarmNetwork(
    swarm_name, swarm_description, agents, api_on=True
)

# Run the agent API
agent_api.run()
