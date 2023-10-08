import discord
from discord.ext import commands

import os
from swarms.agents import Worker
from swarms.agents.memory import VectorStoreRetriever
from swarms.tools.autogpt import WebpageQATool

# Discord bot setup
intents = discord.Intents.default()
bot = commands.Bot(command_prefix='!', intents=intents)

# Memory setup
vectorstore_client = VectorStoreClient() 
retriever = VectorStoreRetriever(vectorstore_client)

# Tools setup
web_search = WebSearchTool(retriever)
memory_tool = MemoryTool(retriever)
tools = [web_search, memory_tool]

# Create the agent
agent = Worker(
  name="DiscordAssistant",
  llm=worker,
  memory=retriever,
  tools=tools
)

@bot.command()
async def query(ctx, *, input):
  response = agent.run(input)
  await ctx.send(response)

bot.run(os.getenv("DISCORD_BOT_TOKEN"))
