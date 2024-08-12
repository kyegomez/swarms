import os
import json
from typing import Dict, List

import requests
from loguru import logger
from swarms.structs.agent import Agent


def add_agent_to_marketplace(
    name: str,
    agent: str,
    language: str,
    description: str,
    use_cases: List[Dict[str, str]],
    requirements: List[Dict[str, str]],
    tags: str,
) -> Dict[str, str]:
    """
    Add an agent to the marketplace.

    Args:
        name (str): The name of the agent.
        agent (str): The agent code.
        language (str): The programming language of the agent.
        description (str): The description of the agent.
        use_cases (List[Dict[str, str]]): The list of use cases for the agent.
        requirements (List[Dict[str, str]]): The list of requirements for the agent.
        tags (str): The tags for the agent.
        api_key (str): The API key for authentication.

    Returns:
        Dict[str, str]: The response from the API.

    Raises:
        requests.exceptions.RequestException: If there is an error making the API request.
    """
    logger.info("Adding agent to marketplace...")

    url = "https://swarms.world/api/add-agent"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv("SWARMS_API_KEY")}",
    }
    data = {
        "name": name,
        "agent": agent,
        "description": description,
        "language": language,
        "useCases": use_cases,
        "requirements": requirements,
        "tags": tags,
    }

    try:
        response = requests.post(
            url, headers=headers, data=json.dumps(data)
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error making API request: {e}")


def add_agent_to_marketplace_sync(
    agent: Agent,
    use_cases: List[Dict[str, str]],
    requirements: List[Dict[str, str]],
    tags: str,
):
    return add_agent_to_marketplace(
        name=agent.agent_name,
        description=agent.description,
        language="python",
        use_cases=use_cases,
        requirements=requirements,
        tags=tags,
    )


# # Example usage
# async def main():
#     name = "Example Agent"
#     agent = "This is an example agent from an API route."
#     description = "Description of the agent."
#     language = "python"
#     use_cases = [
#         {"title": "Use case 1", "description": "Description of use case 1"},
#         {"title": "Use case 2", "description": "Description of use case 2"}
#     ]
#     requirements = [
#         {"package": "pip", "installation": "pip install"},
#         {"package": "pip3", "installation": "pip3 install"}
#     ]
#     tags = "example, agent"
#     api_key = "YOUR_API_KEY"

#     result = await add_agent_to_marketplace(name, agent, language, description, use_cases, requirements, tags, api_key)
#     print(result)

# asyncio.run(main())
