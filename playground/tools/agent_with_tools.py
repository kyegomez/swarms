import os
from swarms.models import OpenAIChat
from swarms.structs import Agent
from swarms.tools.tool import tool
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")


llm = OpenAIChat(api_key=api_key)

# @tool
# def search_api(query: str) -> str:
#     """Search API

#     Args:
#         query (str): _description_

#     Returns:
#         str: _description_
#     """
#     print(f"Searching API for {query}")


## Initialize the workflow
agent = Agent(
    llm=llm,
    max_loops=5,
    # tools=[search_api],
    dashboard=True,
)

out = agent.run(
    "Use the search api to find the best restaurants in New York"
    " City."
)
print(out)
