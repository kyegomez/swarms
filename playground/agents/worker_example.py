import os
from dotenv import load_dotenv
from swarms.agents.worker_agent import Worker
from swarms import OpenAIChat

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

worker = Worker(
    name="My Worker",
    role="Worker",
    human_in_the_loop=False,
    tools=[],
    temperature=0.5,
    llm=OpenAIChat(openai_api_key=api_key),
)

out = worker.run(
    "Hello, how are you? Create an image of how your are doing!"
)
print(out)
