from swarms.models import OpenAIChat
from swarms import Worker
from swarms.prompts import PRODUCT_AGENT_PROMPT

api_key = ""

llm = OpenAIChat(
    openai_api_key=api_key,
    temperature=0.5,
)

node = Worker(
    llm=llm,
    ai_name="Optimus Prime",
    openai_api_key=api_key,
    ai_role=PRODUCT_AGENT_PROMPT,
    external_tools=None,
    human_in_the_loop=False,
    temperature=0.5,
)

task = "Locate 5 trending topics on healthy living, locate a website like NYTimes, and then generate an image of people doing those topics."
response = node.run(task)
print(response)
