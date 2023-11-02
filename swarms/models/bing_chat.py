"""Bing-Chat model by Micorsoft"""
import os
import asyncio
import json
from pathlib import Path

from EdgeGPT.EdgeGPT import Chatbot, ConversationStyle
from EdgeGPT.EdgeUtils import Cookie, ImageQuery, Query
from EdgeGPT.ImageGen import ImageGen


class BingChat:
    """
    EdgeGPT model by OpenAI

    Parameters
    ----------
    cookies_path : str
        Path to the cookies.json necessary for authenticating with EdgeGPT

    Examples
    --------
    >>> edgegpt = BingChat(cookies_path="./path/to/cookies.json")
    >>> response = edgegpt("Hello, my name is ChatGPT")
    >>> image_path = edgegpt.create_img("Sunset over mountains")

    """

    def __init__(self, cookies_path: str = None):
        self.cookies = json.loads(open(cookies_path, encoding="utf-8").read())
        self.bot = asyncio.run(Chatbot.create(cookies=self.cookies))

    def __call__(
        self, prompt: str, style: ConversationStyle = ConversationStyle.creative
    ) -> str:
        """
        Get a text response using the EdgeGPT model based on the provided prompt.
        """
        response = asyncio.run(
            self.bot.ask(
                prompt=prompt, conversation_style=style, simplify_response=True
            )
        )
        return response["text"]

    def create_img(
        self, prompt: str, output_dir: str = "./output", auth_cookie: str = None, auth_cookie_SRCHHPGUSR: str = None
    ) -> str:
        """
        Generate an image based on the provided prompt and save it in the given output directory.
        Returns the path of the generated image.
        """
        if not auth_cookie:
            raise ValueError("Auth cookie is required for image generation.")

        image_generator = ImageGen(auth_cookie, auth_cookie_SRCHHPGUSR, quiet=True, )
        images = image_generator.get_images(prompt)
        image_generator.save_images(images, output_dir=output_dir)

        return Path(output_dir) / images[0]

    @staticmethod
    def set_cookie_dir_path(path: str):
        """
        Set the directory path for managing cookies.
        """
        Cookie.dir_path = Path(path)
