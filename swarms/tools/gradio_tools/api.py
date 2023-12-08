from xml.dom.pulldom import SAX2DOM
from ..tool import Tool
from gradio_tools.tools import (
    BarkTextToSpeechTool,
    StableDiffusionTool,
    DocQueryDocumentAnsweringTool,
    ImageCaptioningTool,
    StableDiffusionPromptGeneratorTool,
    TextToVideoTool,
    ImageToMusicTool,
    WhisperAudioTranscriptionTool,
    ClipInterrogatorTool,
)
from gradio_client.client import Job
from gradio_client.utils import QueueError
import time


def build_tool(config) -> Tool:
    tool = Tool(
        "GradioTool",
        "Gradio_tools Converter",
        name_for_model="gradio_tool",
        description_for_model="Python library for converting Gradio apps into tools that can be leveraged by a large language model (LLM)-based agent to complete its task.",
        logo_url="https://your-app-url.com/.well-known/logo.png",
        contact_email="hello@contact.com",
        legal_info_url="hello@legal.com",
    )

    @tool.get("/get_bark")
    def bark(input: str) -> str:
        """Converting text into sounds that sound like a human read it."""
        bk = BarkTextToSpeechTool()
        return bk.run(input)

    @tool.get("/get_qa")
    def qa(input: str) -> str:
        """Answering questions from the image of the document."""
        qa = DocQueryDocumentAnsweringTool()
        return qa.run(input)

    @tool.get("/get_imagecaption")
    def imagecaption(input: str) -> str:
        """Creating a caption for an image."""
        ic = ImageCaptioningTool()
        return ic.run(input)

    @tool.get("/get_promptgenerator")
    def promptgenerator(input: str) -> str:
        """Generating a prompt for stable diffusion and other image and video generators based on text input."""
        pg = StableDiffusionPromptGeneratorTool()
        return pg.run(input)

    @tool.get("/get_stablediffusion")
    def stablediffusion(input: str) -> str:
        """generate images based on text input."""
        sd = StableDiffusionTool()
        return sd.run(input)

    @tool.get("/get_texttovideo")
    def texttovideo(input: str) -> str:
        """Creating videos from text."""
        tv = TextToVideoTool()
        return tv.run(input)

    @tool.get("/get_imgtomsc")
    def imgtomsc(input: str) -> str:
        """Creating music from images."""
        im = ImageToMusicTool()
        return im.run(input)

    @tool.get("/get_audiotrans")
    def imgtomsc(input: str) -> str:
        """Transcribing an audio file track into text transcript."""
        at = WhisperAudioTranscriptionTool()
        return at.run(input)

    @tool.get("/get_imgprompt")
    def imgprompt(input: str) -> str:
        """Creating a prompt for StableDiffusion that matches the input image."""
        ci = ClipInterrogatorTool()
        return ci.run(input)

    return tool
