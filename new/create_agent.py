import concurrent.futures
from typing import Dict, Optional
import secrets
import string
import uuid

from dotenv import load_dotenv
from swarms import Agent

import replicate

from swarms.utils.str_to_dict import str_to_dict

load_dotenv()


def generate_key(prefix: str = "run") -> str:
    """
    Generates an API key similar to OpenAI's format (sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX).

    Args:
        prefix (str): The prefix for the API key. Defaults to "sk".

    Returns:
        str: An API key string in format: prefix-<48 random characters>
    """
    # Create random string of letters and numbers
    alphabet = string.ascii_letters + string.digits
    random_part = "".join(secrets.choice(alphabet) for _ in range(28))
    return f"{prefix}-{random_part}"


def _generate_media(
    prompt: str = None, modalities: list = None
) -> Dict[str, str]:
    """
    Generate media content (images or videos) based on text prompts using AI models.

    Args:
        prompt (str): Text description of the content to be generated
        modalities (list): List of media types to generate (e.g., ["image", "video"])

    Returns:
        Dict[str, str]: Dictionary containing file paths of generated media
    """
    if not prompt or not modalities:
        raise ValueError("Prompt and modalities must be provided")

    input = {"prompt": prompt}
    results = {}

    def _generate_image(input: Dict) -> str:
        """Generate an image and return the file path."""
        output = replicate.run(
            "black-forest-labs/flux-dev", input=input
        )
        file_paths = []

        for index, item in enumerate(output):
            unique_id = str(uuid.uuid4())
            artifact = item.read()
            file_path = f"output_{unique_id}_{index}.webp"

            with open(file_path, "wb") as file:
                file.write(artifact)

            file_paths.append(file_path)

        return file_paths

    def _generate_video(input: Dict) -> str:
        """Generate a video and return the file path."""
        output = replicate.run("luma/ray", input=input)
        unique_id = str(uuid.uuid4())
        artifact = output.read()
        file_path = f"output_{unique_id}.mp4"

        with open(file_path, "wb") as file:
            file.write(artifact)

        return file_path

    for modality in modalities:
        if modality == "image":
            results["images"] = _generate_image(input)
        elif modality == "video":
            results["video"] = _generate_video(input)
        else:
            raise ValueError(f"Unsupported modality: {modality}")

    print(results)

    return results


def generate_media(
    modalities: list,
    prompt: Optional[str] = None,
    count: int = 1,
) -> Dict:
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=count
    ) as executor:
        # Create list of identical tasks to run concurrently
        futures = [
            executor.submit(
                _generate_media,
                prompt=prompt,  # Fix: Pass as keyword arguments
                modalities=modalities,
            )
            for _ in range(count)
        ]

        # Wait for all tasks to complete and collect results
        results = [
            future.result()
            for future in concurrent.futures.as_completed(futures)
        ]

    return {"results": results}


tools = [
    {
        "type": "function",
        "function": {
            "name": "generate_media",
            "description": "Generate different types of media content (image, video, or music) based on text prompts using AI models.",
            "parameters": {
                "type": "object",
                "properties": {
                    "modality": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["image", "video", "music"],
                        },
                        "description": "The type of media content to generate",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the content to be generated. Specialize it for the modality at hand. For example, if you are generating an image, the prompt should be a description of the image you want to see. If you are generating a video, the prompt should be a description of the video you want to see. If you are generating music, the prompt should be a description of the music you want to hear.",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of outputs to generate (1-4)",
                    },
                },
                "required": [
                    "modality",
                    "prompt",
                    "count",
                ],
            },
        },
    }
]


MEDIA_GENERATION_SYSTEM_PROMPT = """
You are an expert AI Media Generation Assistant, specialized in crafting precise and effective prompts for generating images, videos, and music. Your role is to help users create high-quality media content by understanding their requests and translating them into optimal prompts.

GENERAL GUIDELINES:
- Always analyze the user's request carefully to determine the appropriate modality (image, video, or music)
- Maintain a balanced level of detail in prompts - specific enough to capture the desired outcome but not overly verbose
- Consider the technical limitations and capabilities of AI generation systems
- When unclear, ask for clarification about specific details or preferences

MODALITY-SPECIFIC GUIDELINES:

1. IMAGE GENERATION:
- Structure prompts with primary subject first, followed by style, mood, and technical specifications
- Include relevant art styles when specified (e.g., "digital art", "oil painting", "watercolor", "photorealistic")
- Consider composition elements (foreground, background, lighting, perspective)
- Use specific adjectives for clarity (instead of "beautiful", specify "vibrant", "ethereal", "gritty", etc.)

Example image prompts:
- "A serene Japanese garden at sunset, with cherry blossoms falling, painted in traditional ukiyo-e style, soft pastel colors"
- "Cyberpunk cityscape at night, neon lights reflecting in rain puddles, hyper-realistic digital art style"

2. VIDEO GENERATION:
- Describe the sequence of events clearly
- Specify camera movements if relevant (pan, zoom, tracking shot)
- Include timing and transitions when necessary
- Focus on dynamic elements and motion

Example video prompts:
- "Timelapse of a flower blooming in a garden, close-up shot, soft natural lighting, 10-second duration"
- "Drone shot flying through autumn forest, camera slowly rising above the canopy, revealing mountains in the distance"

3. MUSIC GENERATION:
- Specify genre, tempo, and mood
- Mention key instruments or sounds
- Include emotional qualities and intensity
- Reference similar artists or styles if relevant

Example music prompts:
- "Calm ambient electronic music with soft synthesizer pads, gentle piano melodies, 80 BPM, suitable for meditation"
- "Upbeat jazz fusion track with prominent bass line, dynamic drums, and horn section, inspired by Weather Report"

COUNT HANDLING:
- When multiple outputs are requested (1-4), maintain consistency while introducing subtle variations
- For images: Vary composition or perspective while maintaining style
- For videos: Adjust camera angles or timing while keeping the core concept
- For music: Modify instrument arrangements or tempo while preserving the genre and mood

PROMPT OPTIMIZATION PROCESS:
1. Identify core requirements from user input
2. Determine appropriate modality
3. Add necessary style and technical specifications
4. Adjust detail level based on complexity
5. Consider count and create variations if needed

EXAMPLES OF HANDLING USER REQUESTS:

User: "I want a fantasy landscape"
Assistant response: {
    "modality": "image",
    "prompt": "Majestic fantasy landscape with floating islands, crystal waterfalls, and ancient magical ruins, ethereal lighting, digital art style with rich colors",
    "count": 1
}

User: "Create 3 variations of a peaceful nature scene"
Assistant response: {
    "modality": "image",
    "prompt": "Tranquil forest clearing with morning mist, sunbeams filtering through ancient trees, photorealistic style with soft natural lighting",
    "count": 1
}

IMPORTANT CONSIDERATIONS:
- Avoid harmful, unethical, or inappropriate content
- Respect copyright and intellectual property guidelines
- Maintain consistency with brand guidelines when specified
- Consider technical limitations of current AI generation systems

"""

# Initialize the agent with the new system prompt
agent = Agent(
    agent_name="Media-Generation-Agent",
    agent_description="AI Media Generation Assistant",
    system_prompt=MEDIA_GENERATION_SYSTEM_PROMPT,
    max_loops=1,
    tools_list_dictionary=tools,
    output_type="final",
)


def create_agent(task: str):
    output = str_to_dict(agent.run(task))

    print(output)
    print(type(output))

    prompt = output["prompt"]
    count = output["count"]
    modalities = output["modality"]

    output = generate_media(
        modalities=modalities,
        prompt=prompt,
        count=count,
    )

    run_id = generate_key()

    total_cost = 0

    for modality in modalities:
        if modality == "image":
            total_cost += 0.1
        elif modality == "video":
            total_cost += 1

    result = {
        "id": run_id,
        "success": True,
        "prompt": prompt,
        "count": count,
        "modality": modalities,
        "total_cost": total_cost,
    }

    return result


if __name__ == "__main__":
    task = "Create 3 super kawaii variations of a magical Chinese mountain garden scene in anime style! 🌸✨ Include adorable elements like: cute koi fish swimming in crystal ponds, fluffy clouds floating around misty peaks, tiny pagodas with twinkling lights, and playful pandas hiding in bamboo groves. Make it extra magical with sparkles and soft pastel colors! Create both a video and an image for each variation. Just 1."
    output = create_agent(task)
    print("✨ Yay! Here's your super cute creation! ✨")
    print(output)
