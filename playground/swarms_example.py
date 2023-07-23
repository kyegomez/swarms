from swarms import HierarchicalSwarm
import os

# Retrieve your API key from the environment or replace with your actual key
api_key = ""

# Initialize HierarchicalSwarm with your API key
swarm = HierarchicalSwarm(api_key)

# Define an objective
objective = "Find 20 potential customers for a HierarchicalSwarm based AI Agent automation infrastructure"

# Run HierarchicalSwarm
swarm.run(objective)