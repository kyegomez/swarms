from swarms import worker_node

# Your OpenAI API key
api_key = "sksdsds"

# Initialize a WorkerNode with your API key
node = worker_node(api_key)

# Define an objective
objective = "Please make a web GUI for using HTTP API server..."

# Run the task
task = node.run(objective)

print(task)
