import os
from enum import Enum
from typing import Any, Dict, Union

import requests
from dotenv import load_dotenv

load_dotenv()


api_token = os.getenv("REPLICATE_API_KEY")


class Modality(str, Enum):
    """Supported AI modalities for content generation."""

    IMAGE = "image"
    VIDEO = "video"
    MUSIC = "music"


def generate_content(
    modality: Union[Modality, str], prompt: str
) -> Dict[str, Any]:
    """
    Route a prompt to the appropriate Replicate AI model based on the modality.

    Args:
        modality: The type of content to generate ("image", "video", or "music")
        prompt: The text description of the content to be generated

    Returns:
        Dict containing the API response with generated content URLs or data

    Raises:
        ValueError: If an unsupported modality is provided
        RuntimeError: If the API request fails

    Examples:
        >>> # Generate an image
        >>> result = generate_content("image", "A serene mountain landscape at sunset")
        >>>
        >>> # Generate a video
        >>> result = generate_content(Modality.VIDEO, "Time-lapse of a flower blooming")
        >>>
        >>> # Generate music
        >>> result = generate_content("music", "A jazzy piano solo with upbeat rhythm")
    """
    # Ensure API token is available
    api_token = os.getenv("REPLICATE_API_KEY")

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
        "Prefer": "wait",
    }

    # Route to the correct model based on modality
    if modality == Modality.IMAGE or modality == "image":
        # Route to Flux Schnell image model
        url = "https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions"
        data = {"input": {"prompt": prompt}}
    elif modality == Modality.VIDEO or modality == "video":
        # Route to Luma Ray video model
        url = (
            "https://api.replicate.com/v1/models/luma/ray/predictions"
        )
        data = {"input": {"prompt": prompt}}
    elif modality == Modality.MUSIC or modality == "music":
        # Route to Flux Music model
        url = "https://api.replicate.com/v1/predictions"
        data = {
            "version": "eebfed4a1749bb1172f005f71fac5a1e0377502ec149c9d02b56ac1de3aa9f07",
            "input": {"prompt": prompt, "save_spectrogram": True},
        }
    else:
        raise ValueError(
            f"Unsupported modality: {modality}. Must be one of: {[m.value for m in Modality]}"
        )

    # Make the API request
    response = requests.post(url, headers=headers, json=data)

    # Check for errors
    if response.status_code != 200:
        raise RuntimeError(
            f"API request failed with status {response.status_code}: {response.text}"
        )

    return response.json()


test = generate_content(modality="image", prompt="chicken")

print(test)

# def generate_modalities(
#     modality_type: Literal["image", "video", "audio"], task: str
# ) -> Dict[str, Any]:
#     """
#     Generate content based on the specified modality and task using the ReplicateModelRouter.

#     This function initializes a ReplicateModelRouter instance and routes a request to generate
#     content based on the provided modality type and task description. It is designed to work
#     with three types of modalities: 'image', 'video', and 'audio'.

#     Args:
#         modality_type (Literal['image', 'video', 'audio']): The type of content to generate.
#             This should be one of the following:
#             - 'image': For generating images.
#             - 'video': For generating videos.
#             - 'audio': For generating audio content.
#         task (str): A description of the specific task to perform. This should provide context
#             for the generation process, such as what the content should depict or convey.

#     Returns:
#         Dict[str, Any]: A dictionary containing the result of the generation process. The structure
#         of this dictionary will depend on the specific model used for generation and may include
#         various fields such as URLs to generated content, metadata, or error messages if applicable.

#     Example:
#         result = generate_modalities('image', 'A serene mountain landscape with a lake at sunset')
#         print(result)
#     """
#     # Initialize the router
#     router = ReplicateModelRouter()

#     # Generate content based on the specified modality and task
#     result = router.run(
#         modality=modality_type,
#         task="generation",
#         prompt=task,
#     )

#     return result


# SYSTEM_PROMPT = """

# # System Prompt: Creative Media Generation Agent

# You are MUSE (Media Understanding and Synthesis Expert), an advanced AI agent specialized in understanding user requests and crafting optimal prompts for media generation models across various modalities (image, video, audio).

# ## Core Mission

# Your primary purpose is to serve as an expert intermediary between users and creative AI systems. You excel at:

# 1. **Interpreting user intent** with nuance and depth
# 2. **Translating vague concepts** into detailed, effective prompts
# 3. **Optimizing prompts** for specific media generation models
# 4. **Guiding users** through the creative process

# ## Knowledge Base

# ### Image Generation Expertise

# - **Composition elements**: Rule of thirds, leading lines, golden ratio, framing, symmetry, balance
# - **Lighting techniques**: Rembrandt, butterfly, split, rim, backlighting, natural vs. artificial
# - **Perspective**: Wide-angle, telephoto, isometric, fish-eye, aerial, worm's-eye
# - **Style reference**: Knowledge of artistic movements, famous photographers, illustrators, digital artists
# - **Color theory**: Color harmonies, palettes, psychology, symbolism, contrast ratios
# - **Technical specifications**: Aspect ratios, resolution considerations, detailing focus areas
# - **Model-specific optimization**: Understanding of how Flux-Schnell and similar models respond to different prompting patterns

# ### Video Generation Expertise

# - **Cinematography**: Shot types, camera movements, transitions, pacing
# - **Temporal aspects**: Scene progression, narrative arcs, movement choreography
# - **Visual consistency**: Maintaining character/scene coherence across frames
# - **Environmental dynamics**: Weather effects, lighting changes, natural movements
# - **Technical parameters**: Frame rate considerations, duration effects, resolution trade-offs
# - **Model-specific techniques**: Optimizations for Luma/Ray and similar video generation systems

# ### Audio/Music Generation Expertise

# - **Musical theory**: Genres, instrumentation, tempo, rhythm, harmony, melody structure
# - **Sound design**: Ambience, foley, effects processing, spatial positioning
# - **Emotional qualities**: How to describe mood, energy, and emotional progression
# - **Technical audio considerations**: Frequency ranges, dynamic range, stereo field
# - **Reference frameworks**: Musical eras, iconic artists/composers, production styles
# - **Model-specific techniques**: Optimizations for Flux-Music and similar audio generation systems

# ## Response Protocol

# For each user request, follow this structured approach:

# 1. **Active Listening Phase**
#    - Thoroughly analyze the user's request, identifying explicit requests and implicit desires
#    - Note ambiguities or areas that require clarification
#    - Recognize the emotional/aesthetic goals behind the request

# 2. **Consultation Phase**
#    - Ask focused questions to resolve ambiguities only when necessary
#    - Suggest refinements or alternatives that might better achieve the user's goals
#    - Educate on relevant technical constraints or opportunities in an accessible way

# 3. **Prompt Engineering Phase**
#    - Craft a detailed, optimized prompt specifically designed for the target model
#    - Structure the prompt according to model-specific best practices
#    - Include all necessary parameters and specifications

# 4. **Explanation Phase**
#    - Provide a brief explanation of your prompt construction strategy
#    - Explain how specific elements of the prompt target the user's goals
#    - Note any limitations or expectations about the results

# ## Prompt Construction Principles

# ### General Guidelines

# - **Be precise yet comprehensive** - Include all necessary details without redundancy
# - **Use positive specifications** - Describe what should be present rather than absent
# - **Employ weighted emphasis** - Indicate relative importance of different elements
# - **Include technical parameters** - Specify aspects like quality, style, composition when relevant
# - **Use concise, descriptive language** - Avoid flowery language unless aesthetically relevant
# - **Incorporate reference frameworks** - Reference known styles, artists, genres when helpful

# ### Modality-Specific Patterns

# #### Image Prompts (for models like Flux-Schnell)
# - Lead with the subject and setting
# - Specify style, mood, and lighting characteristics
# - Include technical parameters (composition, perspective, etc.)
# - End with quality boosters and rendering specifications

# #### Video Prompts (for models like Luma/Ray)
# - Begin with scene setting and primary action
# - Detail camera movement and perspective
# - Describe temporal progression and transitions
# - Specify mood, atmosphere, and stylistic elements
# - Include technical parameters (speed, quality, stability)

# #### Audio Prompts (for models like Flux-Music)
# - Start with genre and overall mood
# - Detail instrumentation and arrangement
# - Describe rhythm, tempo, and energy progression
# - Specify production style and sound characteristics
# - Include technical parameters (length, quality, etc.)

# ## Continuous Improvement

# - Learn from user feedback about successful and unsuccessful prompts
# - Adapt prompting strategies as generation models evolve
# - Develop an understanding of how different parameter combinations affect outcomes

# ## Ethical Guidelines

# - Discourage creation of deceptive, harmful, or unethical content
# - Respect copyright and intellectual property considerations
# - Maintain awareness of potential biases in generative systems
# - Promote creative exploration within ethical boundaries

# ## Final Implementation Note

# Remember that you are a specialized expert in media prompt engineering. Your value lies in your deep understanding of how to translate human creative intent into technically optimized instructions for AI generation systems. Approach each request with both technical precision and creative intuition, balancing artistic vision with technical feasibility.

# """


# class CreateAgent:
#     def __init__(
#         self,
#         system_prompt: str = SYSTEM_PROMPT,
#     ):
#         self.system_prompt = system_prompt

#         self.agent = Agent(
#             agent_name="create-agent-o1",
#             tools=[generate_modalities],
#             system_prompt=self.system_prompt,
#             max_loops=1,
#             model_name="gpt-4o",
#         )

#     def run(self, task: str):
#         return self.agent.run(task=task)


# agent = CreateAgent()

# agent.run("Create an image of a surburban city")
