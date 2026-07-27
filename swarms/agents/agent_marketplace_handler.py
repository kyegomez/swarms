"""
Swarms Marketplace integration for agents.

This module owns every piece of marketplace behaviour that used to be inlined
in ``swarms/structs/agent.py`` and split across
``swarms/utils/fetch_prompts_marketplace.py`` and
``swarms/utils/swarms_marketplace_utils.py``:

* auth — resolving and validating ``SWARMS_API_KEY``
* fetching — GET a prompt by UUID or name, returned raw or as
  ``(name, description, prompt)``
* loading — folding a fetched prompt into an agent's system prompt, and
  back-filling the agent's name and description when they were left at their
  defaults
* publishing — POST an agent's prompt, metadata, tags, capabilities, and use
  cases to the marketplace

Both directions require ``SWARMS_API_KEY`` to be set. Get one at
https://swarms.world/platform/api-keys

The class works with or without an agent: fetching and publishing are
classmethods usable standalone, while ``load_prompt`` and ``publish`` read the
owning agent.
"""

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote

import httpx
from loguru import logger

from swarms.schemas.agent_errors import AgentInitializationError

DEFAULT_AGENT_NAME = "swarm-worker-01"
DEFAULT_CATEGORY = "research"
DEFAULT_TIMEOUT = 30.0

MARKETPLACE_BASE_URL = "https://swarms.world/api"
API_KEYS_URL = "https://swarms.world/platform/api-keys"


@lru_cache(maxsize=8)
def _client(timeout: float) -> httpx.Client:
    """
    Return a connection-pooled client for the given timeout.

    Cached so repeated fetches reuse the same TCP connection and TLS session
    instead of paying for a fresh handshake on every call. ``httpx.Client`` is
    thread-safe, so concurrent agents can share one.
    """
    return httpx.Client(timeout=timeout)


class AgentMarketplaceHandler:
    """
    Fetches prompts from, and publishes prompts to, the Swarms Marketplace.

    Args:
        agent: The owning ``Agent``. Read for publishing metadata, written to
            when a marketplace prompt is loaded. Optional — omit it to use the
            handler as a standalone marketplace client.

    Example:
        >>> agent.marketplace.load_prompt("550e8400-e29b-41d4-a716-446655440000")
        >>> agent.marketplace.publish()

        >>> # standalone, no agent required
        >>> AgentMarketplaceHandler.fetch(name="code-review-assistant")
    """

    def __init__(self, agent: Any = None):
        self.agent = agent

    ########################################################
    # Auth & transport
    ########################################################

    @staticmethod
    def check_api_key() -> str:
        """
        Return the Swarms API key from the environment.

        Read fresh every call — caching it means a key exported after import,
        or rotated mid-process, is silently ignored, and one ``os.getenv`` is
        free next to an HTTP round trip.

        Raises:
            ValueError: If SWARMS_API_KEY is unset or empty.
        """
        api_key = os.getenv("SWARMS_API_KEY")

        if api_key is None or api_key.strip() == "":
            raise ValueError(
                "Swarms API key is not set. Please set the SWARMS_API_KEY environment variable. "
                f"You can get your key here: {API_KEYS_URL}"
            )

        return api_key

    @classmethod
    def _headers(cls) -> Dict[str, str]:
        """Authorization headers for every marketplace request."""
        return {
            "Authorization": f"Bearer {cls.check_api_key()}",
            "Content-Type": "application/json",
        }

    ########################################################
    # Fetching
    ########################################################

    @classmethod
    def fetch(
        cls,
        prompt_id: Optional[str] = None,
        name: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        return_params_on: bool = True,
    ) -> Optional[
        Union[
            Dict[str, Any],
            Tuple[Optional[str], Optional[str], Optional[str]],
        ]
    ]:
        """
        Fetch a prompt from the marketplace by UUID or by name.

        GET ``https://swarms.world/api/get-prompts/<uuid-or-url-encoded-name>``

        Args:
            prompt_id: The prompt's UUID. Takes precedence over ``name``.
            name: The prompt's name. URL-encoded automatically.
            timeout: Request timeout in seconds.
            return_params_on: If True, return ``(name, description, prompt)``;
                if False, return the full JSON response.

        Returns:
            The tuple or dict described above, or None if the prompt does not
            exist (404).

        Raises:
            ValueError: If neither ``prompt_id`` nor ``name`` is given.
            httpx.HTTPStatusError: For any error status other than 404.
        """
        if not prompt_id and not name:
            raise ValueError(
                "Either prompt_id or name must be provided"
            )

        # Exactly one of the two identifies the prompt; prompt_id wins.
        url = f"{MARKETPLACE_BASE_URL}/get-prompts/{quote(str(prompt_id or name), safe='')}"

        logger.debug(
            f"Fetching prompt: prompt_id={prompt_id}, name={name} url={url}"
        )

        try:
            response = _client(timeout).get(
                url, headers=cls._headers()
            )

            # 404 => not found
            if response.status_code == 404:
                return None

            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error fetching prompt: {e} Response body: {e.response.text}"
            )
            raise
        except Exception as e:
            logger.error(f"Error fetching prompt: {e}")
            raise

        data = response.json()

        if not return_params_on:
            return data

        return (
            data.get("name"),
            data.get("description"),
            data.get("prompt"),
        )

    @classmethod
    def fetch_prompt(
        cls, prompt_id: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Fetch a single prompt by UUID, requiring it to exist.

        Args:
            prompt_id: The marketplace prompt UUID.

        Returns:
            Tuple of ``(name, description, prompt)``.

        Raises:
            ValueError: If no prompt with that ID exists.
        """
        result = cls.fetch(prompt_id=prompt_id)

        if result is None:
            raise ValueError(
                f"Prompt with ID '{prompt_id}' not found in the marketplace. "
                "Please verify the prompt ID is correct."
            )

        return result

    def load_prompt(self, prompt_id: Optional[str] = None) -> None:
        """
        Load a marketplace prompt into the agent.

        Appends the prompt body to the agent's system prompt, and back-fills
        ``agent_name`` / ``agent_description`` when they are still at their
        defaults, so a marketplace prompt can identify the agent it configures.

        Args:
            prompt_id: The marketplace prompt UUID. Defaults to the agent's
                configured ``marketplace_prompt_id``.

        Raises:
            ValueError: If the prompt cannot be found in the marketplace.
            Exception: If the API request fails.
        """
        agent = self.agent
        prompt_id = prompt_id or agent.marketplace_prompt_id

        try:
            name, description, prompt = self.fetch_prompt(prompt_id)
        except Exception as e:
            logger.error(
                f"Error loading prompt '{prompt_id}' from marketplace: {e}"
            )
            raise

        if prompt:
            agent.system_prompt += prompt

        # Back-fill identity only where the agent left the defaults in place
        if name and agent.agent_name == DEFAULT_AGENT_NAME:
            agent.agent_name = agent.name = name

        if description and agent.agent_description is None:
            agent.agent_description = agent.description = description

        logger.info(f"Loaded prompt '{name}' from marketplace")

        if agent.print_on:
            agent.pretty_print(
                f"[Marketplace] Loaded prompt '{name}' from Swarms Marketplace",
                loop_count=0,
            )

    ########################################################
    # Publishing
    ########################################################

    @classmethod
    def add_prompt(
        cls,
        name: str = None,
        prompt: str = None,
        description: str = None,
        use_cases: List[Dict[str, str]] = None,
        tags: str = None,
        is_free: bool = True,
        price_usd: float = 0.0,
        category: str = DEFAULT_CATEGORY,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> Dict[str, Any]:
        """
        Add a prompt to the Swarms marketplace.

        POST ``https://swarms.world/api/add-prompt``

        Args:
            name: The name of the prompt.
            prompt: The prompt text/template.
            description: A description of what the prompt does.
            use_cases: List of dicts with 'title' and 'description' keys.
            tags: Comma-separated string of tags.
            is_free: Whether the prompt is free or paid.
            price_usd: Price in USD (ignored when ``is_free``).
            category: Category of the prompt (e.g. "content", "coding").
            timeout: Request timeout in seconds.

        Returns:
            The parsed API response, or its raw text if it is not JSON.

        Raises:
            ValueError: If any required field is missing.
            httpx.HTTPStatusError: If the API rejects the request.
        """
        required = {
            "name": name,
            "prompt": prompt,
            "description": description,
            "category": category,
            "use_cases": use_cases,
        }

        for field, value in required.items():
            if value is None:
                raise ValueError(f"{field} is required")

        payload = {
            "name": name,
            "prompt": prompt,
            "description": description,
            "useCases": use_cases or [],
            "tags": tags or "",
            "is_free": is_free,
            "price_usd": price_usd,
            "category": category,
        }

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    f"{MARKETPLACE_BASE_URL}/add-prompt",
                    json=payload,
                    headers=cls._headers(),
                )

            # Try to get response body for better error messages
            try:
                body = response.json()
            except Exception:
                body = response.text

            if response.status_code >= 400:
                logger.error(
                    f"Error adding prompt to marketplace: "
                    f"HTTP {response.status_code}: {response.reason_phrase}\nResponse: {body}"
                )

                if (
                    response.status_code in (401, 500)
                    and "auth" in str(body).lower()
                ):
                    logger.error(
                        "Authentication failed. Please check:\n"
                        "1. Your SWARMS_API_KEY environment variable is set correctly\n"
                        "2. Your API key is valid and not expired\n"
                        f"3. You can verify your key at: {API_KEYS_URL}"
                    )

            response.raise_for_status()

            logger.info(
                f"Prompt Name: {name} Successfully added to marketplace"
            )

            return body
        except Exception as e:
            logger.error(f"Error adding prompt to marketplace: {e}")
            raise

    def build_tags(self) -> str:
        """
        Combine the agent's tags and capabilities into one comma-separated string.

        Either list may be empty or unset; only what exists is included.
        """
        agent = self.agent
        combined = (agent.tags or []) + (agent.capabilities or [])

        return ", ".join(str(item) for item in combined)

    def publish(
        self, category: str = DEFAULT_CATEGORY
    ) -> Dict[str, Any]:
        """
        Publish the agent's prompt and metadata to the marketplace.

        Args:
            category: Marketplace category to publish under.

        Returns:
            The marketplace API response.

        Raises:
            AgentInitializationError: If ``use_cases`` was not provided.
        """
        agent = self.agent

        if agent.use_cases is None:
            raise AgentInitializationError(
                "Use cases are required when publishing to the marketplace. "
                "The schema is a list of dictionaries with 'title' and 'description' keys."
            )

        return self.add_prompt(
            name=agent.agent_name,
            prompt=agent.short_memory.get_str(),
            description=agent.agent_description,
            tags=self.build_tags(),
            category=category,
            use_cases=(agent.use_cases if agent.use_cases else None),
        )
