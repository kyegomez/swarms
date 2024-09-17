import json
import os
from difflib import SequenceMatcher

import requests
from dotenv import load_dotenv
from loguru import logger
from supabase import Client, create_client

load_dotenv()

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Swarms API URL and headers
SWARMS_API_URL = "https://swarms.world/api/add-prompt"
SWARMS_API_KEY = os.getenv("SWARMS_API_KEY")
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {SWARMS_API_KEY}",
}

# Configure logger
logger.add(
    "fetch_and_publish_prompts.log", rotation="1 MB"
)  # Log file with rotation


def fetch_and_publish_prompts():
    logger.info("Starting to fetch and publish prompts.")

    # Fetch data from Supabase
    try:
        response = (
            supabase.table("swarms_framework_schema")
            .select("*")
            .execute()
        )
        rows = response.data
        logger.info(f"Fetched {len(rows)} rows from Supabase.")
    except Exception as e:
        logger.error(f"Failed to fetch data from Supabase: {e}")
        return

    # Track published prompts to avoid duplicates
    published_prompts = set()

    for row in rows:
        # Extract agent_name and system_prompt
        data = row.get("data", {})
        agent_name = data.get("agent_name")
        system_prompt = data.get("system_prompt")

        # Skip if either is missing or duplicate
        if not agent_name or not system_prompt:
            logger.warning(
                f"Skipping row due to missing agent_name or system_prompt: {row}"
            )
            continue
        if is_duplicate(system_prompt, published_prompts):
            logger.info(
                f"Skipping duplicate prompt for agent: {agent_name}"
            )
            continue

        # Create the data payload for the marketplace
        prompt_data = {
            "name": f"{agent_name} - System Prompt",
            "prompt": system_prompt,
            "description": f"System prompt for agent {agent_name}.",
            "useCases": extract_use_cases(system_prompt),
            "tags": "agent, system-prompt",
        }

        # Publish to the marketplace
        try:
            response = requests.post(
                SWARMS_API_URL,
                headers=headers,
                data=json.dumps(prompt_data),
            )
            if response.status_code == 200:
                logger.info(
                    f"Successfully published prompt for agent: {agent_name}"
                )
                published_prompts.add(system_prompt)
            else:
                logger.error(
                    f"Failed to publish prompt for agent: {agent_name}. Response: {response.text}"
                )
        except Exception as e:
            logger.error(
                f"Exception occurred while publishing prompt for agent: {agent_name}. Error: {e}"
            )


def is_duplicate(new_prompt, published_prompts):
    """Check if the prompt is a duplicate using semantic similarity."""
    for prompt in published_prompts:
        similarity = SequenceMatcher(None, new_prompt, prompt).ratio()
        if (
            similarity > 0.9
        ):  # Threshold for considering prompts as duplicates
            return True
    return False


def extract_use_cases(prompt):
    """Extract use cases from the prompt by chunking it into meaningful segments."""
    # This is a simple placeholder; you can use a more advanced method to extract use cases
    chunks = [prompt[i : i + 50] for i in range(0, len(prompt), 50)]
    return [
        {"title": f"Use case {idx+1}", "description": chunk}
        for idx, chunk in enumerate(chunks)
    ]


# Main execution
fetch_and_publish_prompts()
