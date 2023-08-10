# import os
# from swarms.swarms.swarms import HierarchicalSwarm

# api_key = os.getenv("OPENAI_API_KEY")

# # Initialize Swarms with your API key
# swarm = HierarchicalSwarm(openai_api_key=api_key)

# # Define an objective
# objective = """
# Please make a web GUI for using HTTP API server. 
# The name of it is Swarms. 
# You can check the server code at ./main.py. 
# The server is served on localhost:8000. 
# Users should be able to write text input as 'query' and url array as 'files', and check the response. 
# Users input form should be delivered in JSON format. 
# I want it to have neumorphism-style. Serve it on port 4500.

# """

# # Run Swarms
# task = swarm.run(objective)

# print(task)




########## V2
# from swarms.agents.models.openai import OpenAI

# chat = OpenAI()
# response = chat("Hello World")
# print(response)



############# v3
import os 
from swarms.workers.vortex_worker import VortexWorkerAgent

api_key = os.getenv("OPENAI_API_KEY")

agent = VortexWorkerAgent(openai_api_key=api_key)

agent.run("Help me find resources about renewable energy.")
