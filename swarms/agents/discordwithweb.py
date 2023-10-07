import discord
from discord.ext import commands

import os
import openai
import requests

from swarms.agents import Agent
from swarms.agents.memory import VectorStoreRetriever
from swarms.tools.base import BaseTool

"""
Custom tools for web search and memory retrieval.

This code snippet defines two custom tools, `WebSearchTool` and `MemoryTool`, which are subclasses of the `BaseTool` class. The `WebSearchTool` makes a request to a search engine API and extracts the text from the top result. The `MemoryTool` retrieves relevant documents from a vector store and extracts the text from the document.

Example Usage:
```python
web_search = WebSearchTool()
result = web_search.run("python programming")
print(result)
# Output: The text from the top search result for "python programming"

memory_tool = MemoryTool(retriever)
result = memory_tool.run("python programming")
print(result)
# Output: The text from the relevant document retrieved from the vector store for "python programming"
```

Inputs:
- `query` (str): The search query or document retrieval query.

Flow:
1. The `WebSearchTool` makes a request to the Bing search engine API with the provided query.
2. The API response is stored in the `response` variable.
3. The text from the top search result is extracted from the API response and stored in the `text` variable.
4. The `MemoryTool` retrieves relevant documents from the vector store using the provided query.
5. The relevant document is stored in the `relevant_doc` variable.
6. The text from the relevant document is extracted and stored in the `text` variable.

Outputs:
- `text` (str): The extracted text from the top search result or relevant document.
"""

# Custom tools
class WebSearchTool(BaseTool):

    def run(self, query: str) -> str:
        
        # Make request to search engine API
        response = requests.get(
            "https://api.bing.com/v7.0/search",
            params={
                "q": query, 
                "count": 1
            }, 
            headers={
                "Ocp-Apim-Subscription-Key": "YOUR_API_KEY"
            }
        )

        # Extract text from top result
        top_result = response.json()["webPages"]["value"][0]
        return top_result["snippet"]


class MemoryTool(BaseTool):

    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, query: str) -> str:
        
        # Retrieve relevant document from vectorstore
        docs = self.retriever.retrieve(query)
        relevant_doc = docs[0]
        
        # Extract text from document
        text = relevant_doc.text
        
        return text

# Discord bot setup
intents = discord.Intents.default()
bot = commands.Bot(command_prefix='!', intents=intents)

# OpenAI API setup
openai.api_key = os.getenv("OPENAI_API_KEY")

# Memory setup
vectorstore_client = VectorStoreClient() 
retriever = VectorStoreRetriever(vectorstore_client)

# Tools setup
web_search = WebSearchTool()
memory = MemoryTool(retriever)
tools = [web_search, memory]

# Create the agent
agent = Agent(
  name="DiscordAssistant",
  llm=openai,
  memory=retriever,
  tools=tools
)

@bot.command()
async def query(ctx, *, input):
  response = agent.run(input)
  await ctx.send(response)
  
bot.run(os.getenv("DISCORD_BOT_TOKEN"))
