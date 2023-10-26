from swarms.models.bing_chat import BingChat
from swarms.workers.worker import Worker
from swarms.tools.autogpt import EdgeGPTTool, tool
from swarms.models import OpenAIChat
import os

api_key = os.getenv("OPENAI_API_KEY")

# Initialize the EdgeGPTModel
edgegpt = BingChat()


# Initialize the Worker with the custom tool
worker = Worker(llm=llm, ai_name="EdgeGPT Worker", external_tools=[edgegpt])

# Use the worker to process a task
task = "Hello, my name is ChatGPT"
response = worker.run(task)
print(response)
