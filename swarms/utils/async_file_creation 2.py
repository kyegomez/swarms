# In order to accelerate the ops of creating files, we use the async file creation method.
import os
import asyncio
from aiofiles import open as aio_open
from typing import List


async def async_create_file(file_path: str, content: str) -> None:
    async with aio_open(file_path, "w") as file:
        await file.write(content)


async def create_multiple_files(
    file_paths: List[str], contents: List[str]
) -> None:
    tasks = [
        async_create_file(
            (file_path, content)
            for file_path, content in zip(file_paths, contents)
        )
    ]
    await asyncio.gather(*tasks)


async def create_file_with_directory(
    file_path: str, content: str
) -> None:
    """
    Creates a file with the specified directory path and content.

    Args:
        file_path (str): The path of the file to be created.
        content (str): The content to be written to the file.

    Returns:
        None
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    await async_create_file(file_path, content)
