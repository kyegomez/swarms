from swarms.models import OpenAIChat
from apps.discord import Bot

llm = OpenAIChat(
    openai_api_key="Enter in your key",
    temperature=0.5,
)

bot = Bot(llm=llm)
task = "What were the winning boston marathon times for the past 5 years (ending in 2022)? Generate a table of the year, name, country of origin, and times."

bot.send_text(task)
bot.run()
