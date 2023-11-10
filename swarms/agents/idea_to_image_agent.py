import os
import logging
from dataclasses import dataclass
from swarms.models.dalle3 import Dalle
from swarms.models import OpenAIChat


@dataclass
class Idea2Image:
    """
    A class used to generate images from text prompts using DALLE-3.

    ...

    Attributes
    ----------
    image : str
        Text prompt for the image to generate
    openai_api_key : str
        OpenAI API key
    cookie : str
        Cookie value for DALLE-3
    output_folder : str
        Folder to save the generated images

    Methods
    -------
    llm_prompt():
        Returns a prompt for refining the image generation
    generate_image():
        Generates and downloads the image based on the prompt


    Usage:
    ------
    from dalle3 import Idea2Image

    idea2image = Idea2Image(
        image="Fish hivemind swarm in light blue avatar anime in zen garden pond concept art anime art, happy fish, anime scenery"
    )
    idea2image.run()
    """

    image: str
    openai_api_key: str = os.getenv("OPENAI_API_KEY") or None
    cookie: str = os.getenv("BING_COOKIE") or None
    output_folder: str = "images/"

    def __post_init__(self):
        self.llm = OpenAIChat(openai_api_key=self.openai_api_key)
        self.dalle = Dalle(self.cookie)

    def llm_prompt(self):
        LLM_PROMPT = f"""
        Refine the USER prompt to create a more precise image tailored to the user's needs using
        an image generator like DALLE-3.

        ###### FOLLOW THE GUIDE BELOW TO REFINE THE PROMPT ######

        - Use natural language prompts up to 400 characters to describe the image you want to generate. Be as specific or vague as needed.

        - Frame your photographic prompts like camera position, lighting, film type, year, usage context. This implicitly suggests image qualities.

        - For illustrations, you can borrow photographic terms like "close up" and prompt for media, style, artist, animation style, etc.

        - Prompt hack: name a film/TV show genre + year to "steal the look" for costumes, lighting, etc without knowing technical details.

        - Try variations of a prompt, make edits, and do recursive uncropping to create interesting journeys and zoom-out effects.

        - Use an image editor like Photopea to uncrop DALL-E outputs and prompt again to extend the image.

        - Combine separate DALL-E outputs into panoramas and murals with careful positioning/editing.

        - Browse communities like Reddit r/dalle2 to get inspired and share your creations. See tools, free image resources, articles.

        - Focus prompts on size, structure, shape, mood, aesthetics to influence the overall vibe and composition.

        - Be more vague or detailed as needed - DALL-E has studied over 400M images and can riff creatively or replicate specific styles.

        - Be descriptive, describe the art style at the end like fusing concept art with anime art or game art or product design art.

        ###### END OF GUIDE ######

        Prompt to refine: {self.image}
        """
        return LLM_PROMPT

    def run(self):
        """
        Generates and downloads the image based on the prompt.

        This method refines the prompt using the llm, opens the website with the query,
        gets the image URLs, and downloads the images to the specified folder.
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO)

        # Refine the prompt using the llm
        image = self.llm_prompt()
        refined_prompt = self.llm(image)
        print(f"Refined prompt: {refined_prompt}")

        # Open the website with your query
        self.dalle.create(refined_prompt)

        # Get the image URLs
        urls = self.dalle.get_urls()

        # Download the images to your specified folder
        self.dalle.download(urls, self.output_folder)
