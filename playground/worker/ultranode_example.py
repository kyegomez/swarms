from swarms import WorkerUltraUltraNode

# Define an objective
objective = """
Please make a web GUI for using HTTP API server.
The name of it is Swarms.
You can check the server code at ./main.py.
The server is served on localhost:8000.
Users should be able to write text input as 'query' and url array as 'files', and check the response.
Users input form should be delivered in JSON format.
I want it to have neumorphism-style. Serve it on port 4500.
"""

node = WorkerUltraUltraNode(objective)
result = node.execute()
