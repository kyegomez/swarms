from swarms.models import OpenAIChat
from swarms import Worker

llm = OpenAIChat(
    openai_api_key="",
    temperature=0.5,
)

node = Worker(
    llm=llm,
    ai_name="Optimus Prime",
    openai_api_key="",
    ai_role="Worker in a swarm",
    external_tools=None,
    human_in_the_loop=False,
    temperature=0.5,
)

task = "What were the winning boston marathon times for the past 5 years (ending in 2022)? Generate a table of the year, name, country of origin, and times."
response = node.run(task)
print(response)
