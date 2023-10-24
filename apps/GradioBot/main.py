import asyncio
import argparse
from collections import Counter
import json
import pathlib
import re


import discord
from discord.ext import commands
import gradio as gr
from gradio import utils
import requests

from typing import Dict, List

from utils import *


lock = asyncio.Lock()

bot = commands.Bot("", intents=discord.Intents(messages=True, guilds=True))


GUILD_SPACES_FILE = "guild_spaces.pkl"


if pathlib.Path(GUILD_SPACES_FILE).exists():
    guild_spaces = read_pickle_file(GUILD_SPACES_FILE)
    assert isinstance(guild_spaces, dict), f"{GUILD_SPACES_FILE} in invalid format."
    guild_blocks = {}
    delete_keys = []
    for k, v in guild_spaces.items():
        try:
            guild_blocks[k] = gr.Interface.load(v, src="spaces")
        except ValueError:
            delete_keys.append(k)
    for k in delete_keys:
        del guild_spaces[k]
else:
    guild_spaces: Dict[int, str] = {}
    guild_blocks: Dict[int, gr.Blocks] = {}


HASHED_USERS_FILE = "users.pkl"

if pathlib.Path(HASHED_USERS_FILE).exists():
    hashed_users = read_pickle_file(HASHED_USERS_FILE)
    assert isinstance(hashed_users, list), f"{HASHED_USERS_FILE} in invalid format."
else:
    hashed_users: List[str] = []


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    print(f"Running in {len(bot.guilds)} servers...")


async def run_prediction(space: gr.Blocks, *inputs):
    inputs = list(inputs)
    fn_index = 0
    processed_inputs = space.serialize_data(fn_index=fn_index, inputs=inputs)
    batch = space.dependencies[fn_index]["batch"]

    if batch:
        processed_inputs = [[inp] for inp in processed_inputs]

    outputs = await space.process_api(
        fn_index=fn_index, inputs=processed_inputs, request=None, state={}
    )
    outputs = outputs["data"]

    if batch:
        outputs = [out[0] for out in outputs]

    processed_outputs = space.deserialize_data(fn_index, outputs)
    processed_outputs = utils.resolve_singleton(processed_outputs)

    return processed_outputs


async def display_stats(message: discord.Message):
    await message.channel.send(
        f"Running in {len(bot.guilds)} servers\n"
        f"Total # of users: {len(hashed_users)}\n"
        f"------------------"
    )
    await message.channel.send(f"Most popular spaces:")
    # display the top 10 most frequently occurring strings and their counts
    spaces = guild_spaces.values()
    counts = Counter(spaces)
    for space, count in counts.most_common(10):
        await message.channel.send(f"- {space}: {count}")


async def load_space(guild: discord.Guild, message: discord.Message, content: str):
    iframe_url = (
        requests.get(f"https://huggingface.co/api/spaces/{content}/host")
        .json()
        .get("host")
    )
    if iframe_url is None:
        return await message.channel.send(
            f"Space: {content} not found. If you'd like to make a prediction, enclose the inputs in quotation marks."
        )
    else:
        await message.channel.send(
            f"Loading Space: https://huggingface.co/spaces/{content}..."
        )
    interface = gr.Interface.load(content, src="spaces")
    guild_spaces[guild.id] = content
    guild_blocks[guild.id] = interface
    asyncio.create_task(update_pickle_file(guild_spaces, GUILD_SPACES_FILE))
    if len(content) > 32 - len(f"{bot.name} []"):  # type: ignore
        nickname = content[: 32 - len(f"{bot.name} []") - 3] + "..."  # type: ignore
    else:
        nickname = content
    nickname = f"{bot.name} [{nickname}]"  # type: ignore
    await guild.me.edit(nick=nickname)
    await message.channel.send(
        "Ready to make predictions! Type in your inputs and enclose them in quotation marks."
    )


async def disconnect_space(bot: commands.Bot, guild: discord.Guild):
    guild_spaces.pop(guild.id, None)
    guild_blocks.pop(guild.id, None)
    asyncio.create_task(update_pickle_file(guild_spaces, GUILD_SPACES_FILE))
    await guild.me.edit(nick=bot.name)  # type: ignore


async def make_prediction(guild: discord.Guild, message: discord.Message, content: str):
    if guild.id in guild_spaces:
        params = re.split(r' (?=")', content)
        params = [p.strip("'\"") for p in params]
        space = guild_blocks[guild.id]
        predictions = await run_prediction(space, *params)
        if isinstance(predictions, (tuple, list)):
            for p in predictions:
                await send_file_or_text(message.channel, p)
        else:
            await send_file_or_text(message.channel, predictions)
        return
    else:
        await message.channel.send(
            "No Space is currently running. Please type in the name of a Hugging Face Space name first, e.g. abidlabs/en2fr"
        )
        await guild.me.edit(nick=bot.name)  # type: ignore


@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return
    h = hash_user_id(message.author.id)
    if h not in hashed_users:
        hashed_users.append(h)
        asyncio.create_task(update_pickle_file(hashed_users, HASHED_USERS_FILE))
    else:
        if message.content:
            content = remove_tags(message.content)
            guild = message.channel.guild
            assert guild, "Message not sent in a guild."

            if content.strip() == "exit":
                await disconnect_space(bot, guild)
            elif content.strip() == "stats":
                await display_stats(message)
            elif content.startswith('"') or content.startswith("'"):
                await make_prediction(guild, message, content)
            else:
                await load_space(guild, message, content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token",
        type=str,
        help="API key for the Discord bot. You can set this to your Discord token if you'd like to make your own clone of the Gradio Bot.",
        required=False,
        default="",
    )
    args = parser.parse_args()

    if args.token.strip():
        discord_token = args.token
        bot.env = "staging"  # type: ignore
        bot.name = "StagingBot"  # type: ignore
    else:
        with open("secrets.json") as fp:
            discord_token = json.load(fp)["discord_token"]
        bot.env = "prod"  # type: ignore
        bot.name = "GradioBot"  # type: ignore

    bot.run(discord_token)
