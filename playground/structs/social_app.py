from ..swarms import HierarchicalSwarm

# Retrieve your API key from the environment or replace with your actual key
api_key = "sksdsds"

# Initialize HierarchicalSwarm with your API key
swarm = HierarchicalSwarm(openai_api_key=api_key)

# Define an objective
objective = """
Please develop and serve a simple community web service.
People can signup, login, post, comment.
Post and comment should be visible at once.
I want it to have neumorphism-style.
The ports you can use are 4500 and 6500.
"""

# Run HierarchicalSwarm
swarm.run(objective)
