import discord
import os
import src.log
import sys
import pkg_resources
import json
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())

# init loggger
logger = src.log.setup_logger(__name__)


def restart_bot():
    # Replace current process with new instance of bot.py
    os.execl(sys.executable, sys.executable, "bot.py")


def check_verion() -> None:
    # Read the requirements.txt file and add each line to a list
    with open("requirements.txt") as f:
        required = f.read().splitlines()

    # For each library listed in requirements.txt, check if the corresponding version is installed
    for package in required:
        # Use the pkg_resources library to get information about the installed version of the library
        package_name, package_verion = package.split("==")
        installed = pkg_resources.get_distribution(package_name)
        # Extract the library name and version number
        name, version = installed.project_name, installed.version
        # Compare the version number to see if it matches the one in requirements.txt
        if package != f"{name}=={version}":
            logger.error(
                f"{name} version {version} is installed but does not match the requirements"
            )
            sys.exit()


@bot.event
async def on_ready():
    bot_status = discord.Status.online
    # bot_activity = discord.Activity(type=discord.ActivityType.playing, name="bing.com")
    await bot.change_presence(status=bot_status, activity=bot_activity)
    for Filename in os.listdir("./cogs"):
        if Filename.endswith(".py"):
            await bot.load_extension(f"cogs.{Filename[:-3]}")
    logger.info(f"{bot.user} is now running!")
    print("Bot is Up and Ready!")
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} commands")
    except Exception as e:
        print(e)


# Load command
@commands.is_owner()
@bot.command()
async def load(ctx, extension):
    await bot.load_extension(f"cogs.{extension}")
    await ctx.author.send(f"> **Loaded {extension} done.**")


# Unload command
@commands.is_owner()
@bot.command()
async def unload(ctx, extension):
    await bot.unload_extension(f"cogs.{extension}")
    await ctx.author.send(f"> **Un-Loaded {extension} done.**")


# Empty discord_bot.log file
@commands.is_owner()
@bot.command()
async def clean(ctx):
    open("discord_bot.log", "w").close()
    await ctx.author.send(f"> **Successfully emptied the file!**")


# Get discord_bot.log file
@commands.is_owner()
@bot.command()
async def getLog(ctx):
    try:
        with open("discord_bot.log", "rb") as f:
            file = discord.File(f)
        await ctx.author.send(file=file)
        await ctx.author.send("> **Send successfully!**")
    except:
        await ctx.author.send("> **Send failed!**")


# Upload new Bing cookies and restart the bot
@commands.is_owner()
@bot.command()
async def upload(ctx):
    if ctx.message.attachments:
        for attachment in ctx.message.attachments:
            if str(attachment)[-4:] == ".txt":
                content = await attachment.read()
                with open("cookies.json", "w", encoding="utf-8") as f:
                    json.dump(json.loads(content), f, indent=2)
                if not isinstance(ctx.channel, discord.abc.PrivateChannel):
                    await ctx.message.delete()
                await ctx.author.send(f"> **Upload new cookies successfully!**")
                logger.warning("\x1b[31mCookies has been setup successfully\x1b[0m")
                restart_bot()
            else:
                await ctx.author.send("> **Didn't get any txt file.**")
    else:
        await ctx.author.send("> **Didn't get any file.**")


if __name__ == "__main__":
    check_verion()
    bot.run(os.getenv("DISCORD_BOT_TOKEN"))
