import discord
from discord.ext import commands
import asyncio
import os
from dotenv import load_dotenv
from invoke import Executor


class BotCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def greet(self, ctx):
        """greets the user."""
        await ctx.send(f"hello, {ctx.author.name}!")

    @commands.command()
    async def help_me(self, ctx):
        """provides a list of commands and their descriptions."""
        help_text = """
        - `!greet`: greets you.
        - `!run [description]`: generates a video based on the given description.
        - `!help_me`: provides this list of commands and their descriptions.
        """
        await ctx.send(help_text)

    @commands.command()
    async def join(self, ctx):
        """joins the voice channel that the user is in."""
        if ctx.author.voice:
            channel = ctx.author.voice.channel
            await channel.connect()
        else:
            await ctx.send("you are not in a voice channel!")

    @commands.command()
    async def leave(self, ctx):
        """leaves the voice channel that the self.bot is in."""
        if ctx.voice_client:
            await ctx.voice_client.disconnect()
        else:
            await ctx.send("i am not in a voice channel!")

    @commands.command()
    async def listen(self, ctx):
        """starts listening to voice in the voice channel that the bot is in."""
        if ctx.voice_client:
            # create a wavesink to record the audio
            sink = discord.sinks.wavesink("audio.wav")
            # start recording
            ctx.voice_client.start_recording(sink)
            await ctx.send("started listening and recording.")
        else:
            await ctx.send("i am not in a voice channel!")

    @commands.command()
    async def generate_image(self, ctx, *, prompt: str = None, imggen: str = None):
        """generates images based on the provided prompt"""
        await ctx.send(f"generating images for prompt: `{prompt}`...")
        loop = asyncio.get_event_loop()

        # initialize a future object for the dalle instance
        future = loop.run_in_executor(Executor, imggen, prompt)

        try:
            # wait for the dalle request to complete, with a timeout of 60 seconds
            await asyncio.wait_for(future, timeout=300)
            print("done generating images!")

            # list all files in the save_directory
            all_files = [
                os.path.join(root, file)
                for root, _, files in os.walk(os.environ("SAVE_DIRECTORY"))
                for file in files
            ]

            # sort files by their creation time (latest first)
            sorted_files = sorted(all_files, key=os.path.getctime, reverse=True)

            # get the 4 most recent files
            latest_files = sorted_files[:4]
            print(f"sending {len(latest_files)} images to discord...")

            # send all the latest images in a single message
            # storage_service = os.environ("STORAGE_SERVICE") # "https://storage.googleapis.com/your-bucket-name/
            # await ctx.send(files=[storage_service.upload(filepath) for filepath in latest_files])

        except asyncio.timeouterror:
            await ctx.send(
                "the request took too long! it might have been censored or you're out of boosts. please try entering the prompt again."
            )
        except Exception as e:
            await ctx.send(f"an error occurred: {e}")

    @commands.command()
    async def send_text(self, ctx, *, text: str, use_agent: bool = True):
        """sends the provided text to the worker and returns the response"""
        if use_agent:
            response = self.bot.agent.run(text)
        else:
            response = self.bot.llm(text)
        await ctx.send(response)

    @commands.Cog.listener()
    async def on_ready(self):
        print(f"we have logged in as {self.bot.user}")

    @commands.Cog.listener()
    async def on_command_error(self, ctx, error):
        """handles errors that occur while executing commands."""
        if isinstance(error, commands.CommandNotFound):
            await ctx.send("that command does not exist!")
        else:
            await ctx.send(f"an error occurred: {error}")


class Bot:
    def __init__(self, llm, command_prefix="!"):
        load_dotenv()

        intents = discord.Intents.default()
        intents.messages = True
        intents.guilds = True
        intents.voice_states = True
        intents.message_content = True

        # setup
        self.llm = llm
        self.bot = commands.Bot(command_prefix="!", intents=intents)
        self.discord_token = os.getenv("DISCORD_TOKEN")
        self.storage_service = os.getenv("STORAGE_SERVICE")

        # Load the BotCommands cog
        self.bot.add_cog(BotCommands(self.bot))

    def run(self):
        self.bot.run(self.discord_token)
