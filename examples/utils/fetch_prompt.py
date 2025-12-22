from dotenv import load_dotenv

from swarms.utils.fetch_prompts_marketplace import (
    fetch_prompts_from_marketplace,
)

load_dotenv()

if __name__ == "__main__":
    prompt = fetch_prompts_from_marketplace(
        prompt_id="0ff9cc2f-390a-4eb1-9d3d-3a045cd2682e"
    )
    print(prompt)
