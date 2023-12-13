from swarms.models.bing_chat import BingChat
from swarms.workers.worker import Worker
from swarms.tools.autogpt import EdgeGPTTool, tool
from swarms.models import OpenAIChat
import os

api_key = os.getenv("OPENAI_API_KEY")

# Initialize the EdgeGPTModel
edgegpt = BingChat(cookies_path="./cookies.txt")


@tool
def edgegpt(task: str = None):
    """A tool to run infrence on the EdgeGPT Model"""
    return EdgeGPTTool.run(task)


# Initialize the language model,
# This model can be swapped out with Anthropic, ETC, Huggingface Models like Mistral, ETC
llm = OpenAIChat(
    openai_api_key=api_key,
    temperature=0.5,
)

# Initialize the Worker with the custom tool
worker = Worker(
    llm=llm, ai_name="EdgeGPT Worker", external_tools=[edgegpt]
)

# Use the worker to process a task
task = "Hello, my name is ChatGPT"
response = worker.run(task)
print(response)
