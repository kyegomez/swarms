import asyncio
from swarms.tools.mcp_client_call import (
    aget_mcp_tools,
    execute_tool_call,
)
import json


async def main():
    tools = await aget_mcp_tools("http://0.0.0.0:8000/sse", "openai")
    print(json.dumps(tools, indent=4))

    # First create the markdown file
    create_result = await execute_tool_call(
        server_path="http://0.0.0.0:8000/sse",
        messages=[
            {
                "role": "user",
                "content": "Create a new markdown file called 'chicken_cat_story'",
            }
        ],
    )
    print("File creation result:", create_result)

    # Then write the story to the file
    story_content = """Title: The Adventures of Clucky and Whiskers

Once upon a time in a quiet, sunlit farm, there lived a curious chicken named Clucky and a mischievous cat named Whiskers. Clucky was known for her vibrant spirit and insatiable curiosity, roaming the farmyard with her head held high. Whiskers, on the other hand, was a clever little feline who always found himself in amusing predicaments; he often ventured into adventures that few dared to imagine.

The unlikely duo first met one fine autumn morning when Whiskers was chasing a playful butterfly near the barn. Clucky, busy pecking at the ground, almost tripped over Whiskers. Apologizing in her gentle clucks, she noticed that Whiskers was not scared at all—instead, he greeted her with a friendly purr. From that day on, the two embarked on countless adventures, exploring every corner of the farm and beyond.

They would roam the rolling meadows, share stories under the starry night sky, and even work together to solve little mysteries that baffled the other animals. Whether it was searching for a hidden pile of treats or finding safe paths through the woods, Clucky and Whiskers proved that friendship can be found in the most unexpected places.

The other animals on the farm watched in amazement as the chicken and the cat not only complemented each other but also became the best of friends. Clucky's boldness and Whiskers' cunning were a perfect match, teaching everyone that differences can create the strongest bonds.

In the heartwarming adventures of Clucky and Whiskers, one could learn that true friendship breaks all barriers, be they of fur or feathers. The legend of the brave chicken and the clever cat lived on forever, reminding everyone on the farm that unity makes life more colorful and joyful.

The End."""

    story_result = await execute_tool_call(
        server_path="http://0.0.0.0:8000/sse",
        messages=[
            {
                "role": "user",
                "content": f"Write this story to the file 'chicken_cat_story.md': {story_content}",
            }
        ],
    )
    print("Story writing result:", story_result)


if __name__ == "__main__":
    asyncio.run(main())
