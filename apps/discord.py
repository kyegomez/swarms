import discord
from discord.ext import commands
from swarms.models import OpenAIChat
from swarms.agents import OmniModalAgent
import os
from dotenv import load_dotenv
from discord.ext import commands

load_dotenv()

intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
intents.voice_states = True
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)
# Setup

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# Initialize the OmniModalAgent
llm = OpenAIChat(model_name="gpt-4")
agent = OmniModalAgent(llm)


@bot.event
async def on_ready():
    print(f"We have logged in as {bot.user}")


@bot.command()
async def greet(ctx):
    """Greets the user."""
    await ctx.send(f"Hello, {ctx.author.name}!")


@bot.command()
async def run(ctx, *, description: str):
    """Generates a video based on the given description."""
    response = agent.run(
        description
    )  # Assuming the response provides information or a link to the generated video
    await ctx.send(response)


@bot.command()
async def help_me(ctx):
    """Provides a list of commands and their descriptions."""
    help_text = """
    - `!greet`: Greets you.
    - `!run [description]`: Generates a video based on the given description.
    - `!help_me`: Provides this list of commands and their descriptions.
    """
    await ctx.send(help_text)

@bot.event
async def on_command_error(ctx, error):
    """Handles errors that occur while executing commands."""
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("That command does not exist!")
    else:
        await ctx.send(f"An error occurred: {error}")

def setup(bot):
    @bot.command()
    async def join(ctx):
        """Joins the voice channel that the user is in."""
        if ctx.author.voice:
            channel = ctx.author.voice.channel
            await channel.connect()
        else:
            await ctx.send("You are not in a voice channel!")

    @bot.command()
    async def leave(ctx):
        """Leaves the voice channel that the bot is in."""
        if ctx.voice_client:
            await ctx.voice_client.disconnect()
        else:
            await ctx.send("I am not in a voice channel!")

# voice_transcription.py
from discord.ext import commands

def setup(bot):
    @bot.command()
    async def listen(ctx):
        """Starts listening to voice in the voice channel that the bot is in."""
        # ... (code for listening to voice and transcribing it goes here)

bot.run("DISCORD_TOKEN")
