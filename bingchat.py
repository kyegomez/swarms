from swarms.models.bing_chat import EdgeGPTModel
from swarms.workers.worker import Worker
from swarms.tools.tool import EdgeGPTTool 

# Initialize the EdgeGPTModel
edgegpt = EdgeGPTModel(cookies_path="./cookies.txt")

# Initialize the custom tool
edgegpt_tool = EdgeGPTTool(edgegpt)

# Initialize the Worker with the custom tool
worker = Worker(
    ai_name="EdgeGPT Worker",
    external_tools=[edgegpt_tool],
)

# Use the worker to process a task
task = "Hello, my name is ChatGPT"
response = worker.run(task)
print(response)
