import os
import asyncio
import dalle3
import discord
import responses
from invoke import Executor
from dotenv import load_dotenv
from discord.ext import commands


class Bot:
    def __init__(self, agent, llm, command_prefix="!"):
        load_dotenv()

        intents = discord.intents.default()
        intents.messages = True
        intents.guilds = True
        intents.voice_states = True
        intents.message_content = True

        # setup
        self.llm = llm
        self.agent = agent
        self.bot = commands.bot(command_prefix="!", intents=intents)
        self.discord_token = os.getenv("DISCORD_TOKEN")
        self.storage_service = os.getenv("STORAGE_SERVICE")

        @self.bot.event
        async def on_ready():
            print(f"we have logged in as {self.bot.user}")

        @self.bot.command()
        async def greet(ctx):
            """greets the user."""
            await ctx.send(f"hello, {ctx.author.name}!")

        @self.bot.command()
        async def help_me(ctx):
            """provides a list of commands and their descriptions."""
            help_text = """
            - `!greet`: greets you.
            - `!run [description]`: generates a video based on the given description.
            - `!help_me`: provides this list of commands and their descriptions.
            """
            await ctx.send(help_text)

        @self.bot.event
        async def on_command_error(ctx, error):
            """handles errors that occur while executing commands."""
            if isinstance(error, commands.commandnotfound):
                await ctx.send("that command does not exist!")
            else:
                await ctx.send(f"an error occurred: {error}")

        @self.bot.command()
        async def join(ctx):
            """joins the voice channel that the user is in."""
            if ctx.author.voice:
                channel = ctx.author.voice.channel
                await channel.connect()
            else:
                await ctx.send("you are not in a voice channel!")

        @self.bot.command()
        async def leave(ctx):
            """leaves the voice channel that the self.bot is in."""
            if ctx.voice_client:
                await ctx.voice_client.disconnect()
            else:
                await ctx.send("i am not in a voice channel!")

        # voice_transcription.py
        @self.bot.command()
        async def listen(ctx):
            """starts listening to voice in the voice channel that the bot is in."""
            if ctx.voice_client:
                # create a wavesink to record the audio
                sink = discord.sinks.wavesink("audio.wav")
                # start recording
                ctx.voice_client.start_recording(sink)
                await ctx.send("started listening and recording.")
            else:
                await ctx.send("i am not in a voice channel!")

        # image_generator.py
        @self.bot.command()
        async def generate_image(ctx, *, prompt: str):
            """generates images based on the provided prompt"""
            await ctx.send(f"generating images for prompt: `{prompt}`...")
            loop = asyncio.get_event_loop()

            # initialize a future object for the dalle instance
            model_instance = dalle3()
            future = loop.run_in_executor(Executor, model_instance.run, prompt)

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
                storage_service = os.environ(
                    "STORAGE_SERVICE"
                )  # "https://storage.googleapis.com/your-bucket-name/
                await ctx.send(
                    files=[
                        storage_service.upload(filepath) for filepath in latest_files
                    ]
                )

            except asyncio.timeouterror:
                await ctx.send(
                    "the request took too long! it might have been censored or you're out of boosts. please try entering the prompt again."
                )
            except Exception as e:
                await ctx.send(f"an error occurred: {e}")

        @self.bot.command()
        async def send_text(ctx, *, text: str, use_agent: bool = True):
            """sends the provided text to the worker and returns the response"""
            if use_agent:
                response = self.agent.run(text)
            else:
                response = self.llm.run(text)
            await ctx.send(response)

        def add_command(self, name, func):
            @self.bot.command()
            async def command(ctx, *args):
                reponse = func(*args)
                await ctx.send(responses)


def run(self):
    self.bot.run("DISCORD_TOKEN")
