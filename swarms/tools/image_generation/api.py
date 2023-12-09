import requests
import json
from ..tool import Tool
import os

from steamship import Block, Steamship
import uuid
from enum import Enum
import re
from IPython import display
from IPython.display import Image


class ModelName(str, Enum):
    """Supported Image Models for generation."""

    DALL_E = "dall-e"
    STABLE_DIFFUSION = "stable-diffusion"


SUPPORTED_IMAGE_SIZES = {
    ModelName.DALL_E: ("256x256", "512x512", "1024x1024"),
    ModelName.STABLE_DIFFUSION: ("512x512", "768x768"),
}


def make_image_public(client: Steamship, block: Block) -> str:
    """Upload a block to a signed URL and return the public URL."""
    try:
        from steamship.data.workspace import SignedUrl
        from steamship.utils.signed_urls import upload_to_signed_url
    except ImportError:
        raise ValueError(
            "The make_image_public function requires the steamship"
            " package to be installed. Please install steamship"
            " with `pip install --upgrade steamship`"
        )

    filepath = str(uuid.uuid4())
    signed_url = (
        client.get_workspace()
        .create_signed_url(
            SignedUrl.Request(
                bucket=SignedUrl.Bucket.PLUGIN_DATA,
                filepath=filepath,
                operation=SignedUrl.Operation.WRITE,
            )
        )
        .signed_url
    )
    read_signed_url = (
        client.get_workspace()
        .create_signed_url(
            SignedUrl.Request(
                bucket=SignedUrl.Bucket.PLUGIN_DATA,
                filepath=filepath,
                operation=SignedUrl.Operation.READ,
            )
        )
        .signed_url
    )
    upload_to_signed_url(signed_url, block.raw())
    return read_signed_url


def show_output(output):
    """Display the multi-modal output from the agent."""
    UUID_PATTERN = re.compile(
        r"([0-9A-Za-z]{8}-[0-9A-Za-z]{4}-[0-9A-Za-z]{4}-[0-9A-Za-z]{4}-[0-9A-Za-z]{12})"
    )

    outputs = UUID_PATTERN.split(output)
    outputs = [
        re.sub(r"^\W+", "", el) for el in outputs
    ]  # Clean trailing and leading non-word characters

    for output in outputs:
        maybe_block_id = UUID_PATTERN.search(output)
        if maybe_block_id:
            display(Image(Block.get(Steamship(), _id=maybe_block_id.group()).raw()))
        else:
            print(output, end="\n\n")


def build_tool(config) -> Tool:
    tool = Tool(
        "Image Generator",
        "Tool that can generate image based on text description.",
        name_for_model="Image Generator",
        description_for_model=(
            "Useful for when you need to generate an image."
            "Input: A detailed text-2-image prompt describing an image"
            "Output: the UUID of a generated image"
        ),
        logo_url="https://your-app-url.com/.well-known/logo.png",
        contact_email="hello@contact.com",
        legal_info_url="hello@legal.com",
    )

    model_name: ModelName = ModelName.DALL_E  # choose model and image size?
    size: Optional[str] = "512x512"
    return_urls: Optional[bool] = False
    steamship_api_key = os.environ.get("STEAMSHIP_API_KEY", "")
    if steamship_api_key == "":
        raise RuntimeError(
            "STEAMSHIP_API_KEY is not provided. Please sign up for a free account at https://steamship.com/account/api, create a new API key, and add it to environment variables."
        )

    steamship = Steamship(
        api_key=steamship_api_key,
    )

    @tool.get("/generate_image")
    def generate_image(query: str):
        """Generate an image."""
        image_generator = steamship.use_plugin(
            plugin_handle=model_name.value, config={"n": 1, "size": size}
        )

        task = image_generator.generate(text=query, append_output_to_file=True)
        task.wait()
        blocks = task.output.blocks
        output_uiud = blocks[0].id
        if len(blocks) > 0:
            if return_urls:
                output_uiud = make_image_public(steamship, blocks[0])
            # print image?
            # show_output(output_uiud)
            return output_uiud
        raise RuntimeError("Tool unable to generate image!")

    return tool
