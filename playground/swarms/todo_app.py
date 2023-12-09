from swarms import HierarchicalSwarm


# Retrieve your API key from the environment or replace with your actual key
api_key = "sksdsds"

# Initialize HierarchicalSwarm with your API key
swarm = HierarchicalSwarm(openai_api_key=api_key)

# Define an objective
objective = """
Please develop and serve a simple web TODO app.
The user can list all TODO items and add or delete each TODO item.
I want it to have neumorphism-style.
The ports you can use are 4500 and 6500.

"""

# Run HierarchicalSwarm
swarm.run(objective)
