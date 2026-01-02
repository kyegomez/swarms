"""
Fetch Prompts from the Swarms Marketplace

This module provides utilities for fetching prompts from the Swarms marketplace API.
It supports fetching prompts by either their unique UUID or by their name.

Environment Variables:
    SWARMS_API_KEY: Optional API key for authenticated requests. Required for
                    accessing paid/premium prompt content.

Example:
    >>> from swarms.utils.fetch_prompts_marketplace import fetch_prompts_from_marketplace
    >>> name, description, prompt = fetch_prompts_from_marketplace(name="my-prompt")
    >>> print(f"Prompt: {name}")
"""

import traceback
from typing import Any, Dict, Optional, Tuple, Union
from urllib.parse import quote
from swarms.utils.swarms_marketplace_utils import check_swarms_api_key

import httpx
from loguru import logger


def return_params(
    data: dict,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract and return the core parameters from a prompt data dictionary.

    Parses a dictionary containing prompt information and extracts the name,
    description, and prompt content fields.

    Args:
        data (dict): A dictionary containing prompt data with the following
            expected keys:
            - "name": The name/title of the prompt
            - "description": A description of what the prompt does
            - "prompt": The actual prompt content/template

    Returns:
        Tuple[Optional[str], Optional[str], Optional[str]]: A tuple containing:
            - name (Optional[str]): The prompt name, or None if not present
            - description (Optional[str]): The prompt description, or None if not present
            - prompt (Optional[str]): The prompt content, or None if not present

    Example:
        >>> data = {
        ...     "name": "Code Review Assistant",
        ...     "description": "A prompt for reviewing code",
        ...     "prompt": "Review the following code for bugs..."
        ... }
        >>> name, desc, prompt = return_params(data)
        >>> print(name)
        'Code Review Assistant'
    """
    name = data.get("name")
    description = data.get("description")
    prompt = data.get("prompt")
    return name, description, prompt


def fetch_prompts_from_marketplace(
    prompt_id: Optional[str] = None,
    name: Optional[str] = None,
    timeout: float = 30.0,
    return_params_on: bool = True,
) -> Optional[
    Union[
        Dict[str, Any],
        Tuple[Optional[str], Optional[str], Optional[str]],
    ]
]:
    """
    Fetch a prompt from the Swarms marketplace by UUID or by prompt name.

    Retrieves prompt data from the Swarms marketplace API. You can fetch a prompt
    either by its unique UUID identifier or by its name. At least one of these
    parameters must be provided.

    API Endpoints:
        - GET https://swarms.world/api/get-prompts/<uuid>
        - GET https://swarms.world/api/get-prompts/<prompt-name> (URL-encoded)

    Args:
        prompt_id (Optional[str]): The unique UUID identifier of the prompt.
            Takes precedence over `name` if both are provided. Defaults to None.
        name (Optional[str]): The name of the prompt to fetch. Will be URL-encoded
            automatically. Defaults to None.
        timeout (float): Request timeout in seconds. Defaults to 30.0.
        return_params_on (bool): If True, returns a tuple of (name, description, prompt).
            If False, returns the full JSON response dictionary. Defaults to True.

    Returns:
        Optional[Union[Dict[str, Any], Tuple[Optional[str], Optional[str], Optional[str]]]]:
            - If `return_params_on` is True: A tuple of (name, description, prompt)
            - If `return_params_on` is False: The full JSON response as a dictionary
            - None if the prompt was not found (404 response)

    Raises:
        ValueError: If neither `prompt_id` nor `name` is provided.
        httpx.HTTPStatusError: If the API returns an error status code (except 404).
        httpx.TimeoutException: If the request exceeds the specified timeout.
        Exception: For any other unexpected errors during the request.

    Note:
        Set the `SWARMS_API_KEY` environment variable to access paid/premium
        prompt content. The API key will be included as a Bearer token in
        the Authorization header.

    Example:
        Fetch a prompt by name and get parsed parameters:

        >>> name, description, prompt = fetch_prompts_from_marketplace(
        ...     name="code-review-assistant"
        ... )
        >>> print(f"Name: {name}")
        >>> print(f"Description: {description}")

        Fetch a prompt by UUID and get the full response:

        >>> response = fetch_prompts_from_marketplace(
        ...     prompt_id="550e8400-e29b-41d4-a716-446655440000",
        ...     return_params_on=False
        ... )
        >>> print(response.keys())

        Handle a non-existent prompt:

        >>> result = fetch_prompts_from_marketplace(name="non-existent-prompt")
        >>> if result is None:
        ...     print("Prompt not found")
    """
    if not prompt_id and not name:
        raise ValueError("Either prompt_id or name must be provided")

    api_key = check_swarms_api_key()

    # New endpoint
    base_url = "https://swarms.world/api/get-prompts"
    id_or_name = (
        prompt_id if prompt_id else name
    )  # exactly one is set here
    url = f"{base_url}/{quote(str(id_or_name), safe='')}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        logger.debug(
            f"Fetching prompt: prompt_id={prompt_id}, name={name} url={url}"
        )

        with httpx.Client(timeout=timeout) as client:
            response = client.get(url, headers=headers)

            # 404 => not found
            if response.status_code == 404:
                return None

            response.raise_for_status()

            if return_params_on:
                return return_params(response.json())
            else:
                return response.json()

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching prompt: {e}")
        if e.response is not None:
            try:
                logger.error(
                    f"Error response body: {e.response.json()}"
                )
            except Exception:
                logger.error(
                    f"Error response text: {e.response.text}"
                )
        raise
    except Exception as e:
        logger.error(
            f"Error fetching prompt: {e} Traceback: {traceback.format_exc()}"
        )
        raise
