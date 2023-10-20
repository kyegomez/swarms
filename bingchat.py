from swarms.models.bing_chat import BingChat
from swarms.workers.worker import Worker
from swarms.tools.autogpt import EdgeGPTTool, tool


# Initialize the language model,
# This model can be swapped out with Anthropic, ETC, Huggingface Models like Mistral, ETC
llm = BingChat(cookies_path="./cookies.json")

# Initialize the Worker with the custom tool
worker = Worker(
    llm=llm,
    ai_name="EdgeGPT Worker",
)

# Use the worker to process a task
task = "Hello, my name is ChatGPT"
response = worker.run(task)
print(response)
