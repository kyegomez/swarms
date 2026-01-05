"""
Agent Marketplace Utilities

This module provides utilities for interacting with the Swarms Agent Marketplace API.
It supports searching, listing, fetching details, and installing agents from the marketplace.

Environment Variables:
    SWARMS_API_KEY: Required API key for authenticated requests.

Example:
    >>> from swarms.utils.agent_marketplace import query_agents, install_agent
    >>> agents = query_agents(search="trading", category="finance")
    >>> install_agent(agent_id="abc123", output_dir="./agents")
"""

import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from swarms.utils.swarms_marketplace_utils import check_swarms_api_key


BASE_URL = "https://swarms.world"


def query_agents(
    search: Optional[str] = None,
    category: Optional[str] = None,
    free_only: bool = False,
    limit: int = 20,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """
    Query agents from the Swarms marketplace.

    Args:
        search: Search keyword for matching agent names/descriptions.
        category: Filter by category (case-insensitive).
        free_only: Only return free agents.
        limit: Maximum number of results (1-100, default 20).
        timeout: Request timeout in seconds (default 30.0).

    Returns:
        Dictionary containing the query results with agent data.

    Raises:
        httpx.HTTPError: If the HTTP request fails.
        ValueError: If SWARMS_API_KEY is not set.
    """
    try:
        api_key = check_swarms_api_key()
        url = f"{BASE_URL}/api/query-agents"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "limit": min(max(1, limit), 100),
        }

        if search:
            data["search"] = search
        if category:
            data["category"] = category
        if free_only:
            data["priceFilter"] = "free"

        logger.debug(f"Querying agents with params: {data}")

        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=data, headers=headers)
            response.raise_for_status()
            result = response.json()

        # Parse response - API returns {"data": [...], "total": N}
        agents = result.get("data", [])
        total = result.get("total", len(agents))

        return {"agents": agents, "total": total}

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error querying agents: {e}")
        if hasattr(e, "response") and e.response is not None:
            try:
                error_body = e.response.json()
                logger.error(f"Error response: {error_body}")
            except Exception:
                logger.error(f"Error response text: {e.response.text}")
        raise
    except Exception as e:
        logger.error(
            f"Error querying agents: {e} Traceback: {traceback.format_exc()}"
        )
        raise


def get_agent_by_id(
    agent_id: str,
    timeout: float = 30.0,
) -> Optional[Dict[str, Any]]:
    """
    Fetch a specific agent from the marketplace by its ID.

    Args:
        agent_id: The unique identifier of the agent.
        timeout: Request timeout in seconds (default 30.0).

    Returns:
        Dictionary containing the agent data, or None if not found.

    Raises:
        httpx.HTTPError: If the HTTP request fails.
        ValueError: If SWARMS_API_KEY is not set.
    """
    try:
        # Query with a search that should match the ID
        # Since there's no direct get-by-id endpoint, we query and filter
        result = query_agents(search=agent_id, limit=100, timeout=timeout)

        if isinstance(result, list):
            agents = result
        elif isinstance(result, dict):
            agents = result.get("agents", result.get("data", []))
            if not isinstance(agents, list):
                agents = [result] if result.get("id") == agent_id else []
        else:
            agents = []

        # Find exact match by ID
        for agent in agents:
            if agent.get("id") == agent_id:
                return agent

        # If no exact match found, return first result if search was specific
        if agents and len(agent_id) > 10:  # Likely a UUID
            return agents[0] if agents else None

        return None

    except Exception as e:
        logger.error(f"Error fetching agent {agent_id}: {e}")
        raise


def install_agent(
    agent_id: str,
    output_dir: str = ".",
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """
    Install an agent from the marketplace to a local directory.

    Creates a Python file with the agent configuration and metadata.

    Args:
        agent_id: The unique identifier of the agent to install.
        output_dir: Directory to save the agent file (default: current directory).
        timeout: Request timeout in seconds (default 30.0).

    Returns:
        Dictionary with installation result including file path.

    Raises:
        httpx.HTTPError: If the HTTP request fails.
        ValueError: If SWARMS_API_KEY is not set or agent not found.
        OSError: If file creation fails.
    """
    try:
        # Fetch agent details
        agent = get_agent_by_id(agent_id, timeout=timeout)

        if not agent:
            raise ValueError(f"Agent with ID '{agent_id}' not found in marketplace")

        # Extract agent information
        agent_name = agent.get("name", "marketplace_agent")
        agent_code = agent.get("agent", "")
        description = agent.get("description", "")
        language = agent.get("language", "python")
        requirements = agent.get("requirements", [])
        tags = agent.get("tags", "")
        use_cases = agent.get("useCases", [])

        # Sanitize filename
        safe_name = "".join(
            c if c.isalnum() or c in ("_", "-") else "_"
            for c in agent_name.lower()
        ).strip("_")

        if not safe_name:
            safe_name = "marketplace_agent"

        # Determine file extension based on language
        ext = ".py" if language.lower() == "python" else f".{language.lower()}"
        filename = f"{safe_name}{ext}"

        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        file_path = output_path / filename

        # Build file content with metadata header
        content_lines = [
            '"""',
            f"Agent: {agent_name}",
            f"Description: {description}",
            f"",
            f"Source: Swarms Marketplace",
            f"Agent ID: {agent_id}",
        ]

        if tags:
            content_lines.append(f"Tags: {tags}")

        if requirements:
            # Handle requirements as list of dicts or strings
            req_strs = []
            for req in requirements:
                if isinstance(req, dict):
                    req_strs.append(req.get("package", str(req)))
                else:
                    req_strs.append(str(req))
            content_lines.append(f"Requirements: {', '.join(req_strs)}")

        if use_cases:
            content_lines.append("")
            content_lines.append("Use Cases:")
            for uc in use_cases:
                title = uc.get("title", "")
                desc = uc.get("description", "")
                if title:
                    content_lines.append(f"  - {title}: {desc}")

        content_lines.append('"""')
        content_lines.append("")

        # Add requirements comment if present
        if requirements:
            content_lines.append("# Requirements:")
            for req in requirements:
                if isinstance(req, dict):
                    install_cmd = req.get("installation", f"pip install {req.get('package', '')}")
                    content_lines.append(f"# {install_cmd}")
                else:
                    content_lines.append(f"# pip install {req}")
            content_lines.append("")

        # Add the agent code
        if agent_code:
            content_lines.append(agent_code)
        else:
            # Provide a template if no code is available
            # Escape description for use in triple-quoted string
            safe_description = description.replace('"""', '\\"\\"\\"')
            safe_name = agent_name.replace('"', '\\"')
            content_lines.extend([
                "from swarms import Agent",
                "",
                "agent = Agent(",
                f'    agent_name="{safe_name}",',
                f'    agent_description="""{safe_description}""",',
                '    model_name="gpt-4o-mini",',
                "    max_loops=1,",
                ")",
                "",
                "# Run the agent",
                '# result = agent.run("Your task here")',
            ])

        # Write the file
        file_content = "\n".join(content_lines)
        file_path.write_text(file_content, encoding="utf-8")

        logger.info(f"Agent '{agent_name}' installed to {file_path}")

        return {
            "success": True,
            "agent_name": agent_name,
            "file_path": str(file_path),
            "agent_id": agent_id,
            "description": description,
        }

    except ValueError:
        raise
    except Exception as e:
        logger.error(
            f"Error installing agent: {e} Traceback: {traceback.format_exc()}"
        )
        raise


def list_available_categories(timeout: float = 30.0) -> List[str]:
    """
    Get a list of available agent categories from the marketplace.

    Args:
        timeout: Request timeout in seconds (default 30.0).

    Returns:
        List of category names.
    """
    # Common categories based on marketplace structure
    return [
        "finance",
        "research",
        "coding",
        "content",
        "data-analysis",
        "automation",
        "customer-service",
        "healthcare",
        "legal",
        "marketing",
        "education",
        "general",
    ]
