from discord.ext import commands
from swarms.models import OpenAIChat
from swarms.agents import OmniModalAgent

# Setup
TOKEN = "YOUR_DISCORD_BOT_TOKEN"
bot = commands.Bot(command_prefix="!")

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


bot.run(TOKEN)
