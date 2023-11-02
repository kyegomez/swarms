from tabnanny import verbose
from click import prompt
from langchain import LLMChain
from swarms.models import OpenAIChat
from swarms import Worker
from swarms.prompts import PRODUCT_AGENT_PROMPT
from swarms.models.bing_chat import BingChat

# api_key = ""

# llm = OpenAIChat(
#     openai_api_key=api_key,
#     temperature=0.5,
# )

llm = BingChat(cookies_path="./cookies.json")
# llm = LLMChain(llm=bing.to_dict(), prompt=prompt, verbose=verbose)

node = Worker(
    llm=llm,
    ai_name="Optimus Prime",
    ai_role=PRODUCT_AGENT_PROMPT,
    external_tools=None,
    human_in_the_loop=False,
    temperature=0.5,
    use_openai=False
)

task = "Locate 5 trending topics on healthy living, locate a website like NYTimes, and then generate an image of people doing those topics."
response = node.run(task)
print(response)
