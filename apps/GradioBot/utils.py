from __future__ import annotations

import asyncio
import pickle
import hashlib
import pathlib
from typing import Dict, List

import discord

lock = asyncio.Lock()


async def update_pickle_file(data: Dict | List, file_path: str):
    async with lock:
        with open(file_path, "wb") as fp:
            pickle.dump(data, fp)


def read_pickle_file(file_path: str):
    with open(file_path, "rb") as fp:
        return pickle.load(fp)


async def send_file_or_text(channel, file_or_text: str):
    # if the file exists, send as a file
    if pathlib.Path(str(file_or_text)).exists():
        with open(file_or_text, "rb") as f:
            return await channel.send(file=discord.File(f))
    else:
        return await channel.send(file_or_text)


def remove_tags(content: str) -> str:
    content = content.replace("<@1040198143695933501>", "")
    content = content.replace("<@1057338428938788884>", "")
    return content.strip()


def hash_user_id(user_id: int) -> str:
    return hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()
