from swarms import Swarms
import os

# Retrieve your API key from the environment or replace with your actual key
api_key = ""

# Initialize Swarms with your API key
swarm = Swarms(api_key)

# Define an objective
objective = "Find 20 potential customers for a Swarms based AI Agent automation infrastructure"

# Run Swarms
swarm.run_swarms(objective)