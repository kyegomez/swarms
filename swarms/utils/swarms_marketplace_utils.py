import os
import traceback
from typing import Any, Dict, List

import httpx
from loguru import logger


def add_prompt_to_marketplace(
    name: str = None,
    prompt: str = None,
    description: str = None,
    use_cases: List[Dict[str, str]] = None,
    tags: str = None,
    is_free: bool = True,
    price_usd: float = 0.0,
    category: str = "research",
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """
    Add a prompt to the Swarms marketplace.

    Args:
        name: The name of the prompt.
        prompt: The prompt text/template.
        description: A description of what the prompt does.
        use_cases: List of dictionaries with 'title' and 'description' keys
            describing use cases for the prompt.
        tags: Comma-separated string of tags for the prompt.
        is_free: Whether the prompt is free or paid.
        price_usd: Price in USD (ignored if is_free is True).
        category: Category of the prompt (e.g., "content", "coding", etc.).
        timeout: Request timeout in seconds. Defaults to 30.0.

    Returns:
        Dictionary containing the API response.

    Raises:
        httpx.HTTPError: If the HTTP request fails.
        httpx.RequestError: If there's an error making the request.
    """
    try:
        url = "https://swarms.world/api/add-prompt"
        api_key = os.getenv("SWARMS_API_KEY")

        if api_key is None or api_key.strip() == "":
            raise ValueError(
                "Swarms API key is not set. Please set the SWARMS_API_KEY environment variable. "
                "You can get your key here: https://swarms.world/platform/api-keys"
            )

        # Log that we have an API key (without exposing it)
        logger.debug(
            f"Using API key (length: {len(api_key)} characters)"
        )

        # Validate required fields
        if name is None:
            raise ValueError("name is required")
        if prompt is None:
            raise ValueError("prompt is required")
        if description is None:
            raise ValueError("description is required")
        if category is None:
            raise ValueError("category is required")
        if use_cases is None:
            raise ValueError("use_cases is required")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "name": name,
            "prompt": prompt,
            "description": description,
            "useCases": use_cases or [],
            "tags": tags or "",
            "is_free": is_free,
            "price_usd": price_usd,
            "category": category,
        }

        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=data, headers=headers)

            # Try to get response body for better error messages
            try:
                response_body = response.json()
            except Exception:
                response_body = response.text

            if response.status_code >= 400:
                error_msg = f"HTTP {response.status_code}: {response.reason_phrase}"
                if response_body:
                    error_msg += f"\nResponse: {response_body}"
                logger.error(
                    f"Error adding prompt to marketplace: {error_msg}"
                )

            response.raise_for_status()
            logger.info(
                f"Prompt Name: {name} Successfully added to marketplace"
            )
            return response_body
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error adding prompt to marketplace: {e}")
        if hasattr(e, "response") and e.response is not None:
            try:
                error_body = e.response.json()
                logger.error(f"Error response body: {error_body}")

                # Provide helpful error message for authentication failures
                if (
                    e.response.status_code == 401
                    or e.response.status_code == 500
                ):
                    if isinstance(error_body, dict):
                        if (
                            "authentication"
                            in str(error_body).lower()
                            or "auth" in str(error_body).lower()
                        ):
                            logger.error(
                                "Authentication failed. Please check:\n"
                                "1. Your SWARMS_API_KEY environment variable is set correctly\n"
                                "2. Your API key is valid and not expired\n"
                                "3. You can verify your key at: https://swarms.world/platform/api-keys"
                            )
            except Exception:
                logger.error(
                    f"Error response text: {e.response.text}"
                )
        raise
    except Exception as e:
        logger.error(
            f"Error adding prompt to marketplace: {e} Traceback: {traceback.format_exc()}"
        )
        raise
