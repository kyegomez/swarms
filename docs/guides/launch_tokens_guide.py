import json
import os
from typing import Optional

import httpx
from dotenv import load_dotenv

from swarms import Agent

load_dotenv()

BASE_URL = "https://swarms.world"

PRIVATE_KEY = os.getenv("PRIVATE_KEY")
SWARMS_API_KEY = os.getenv("SWARMS_API_KEY")

# System prompt: specialize the agent as a meme-coin launch expert
MEME_COIN_LAUNCH_SYSTEM_PROMPT = """
You are an expert at launching meme coins. Your specialty is:

- **Naming & branding**: Create catchy, memorable token names and tickers (short, punchy, often 3–5 chars) that fit the meme or theme.
- **Copy & narrative**: Write sharp, viral-ready descriptions and one-liners that explain the joke or community without sounding corporate.
- **Creative direction**: Suggest or interpret image/logo ideas (URLs or base64) that match the meme aesthetic—bold, simple, recognizable.
- **Launch strategy**: When asked, advise on timing, community hooks, and how to describe the token so it resonates with degens and normies alike.

You have tools: `launch_token(name, description, ticker, image)` to create a token on the Swarms launchpad, and `claim_fees_httpx(contract_address)` to claim fees. Use them when the user wants to launch a token or manage fees. Always confirm key details (name, ticker, description, image) with the user before calling launch_token unless they have already provided everything. Be concise, creative, and on-brand for meme coins.
"""


def launch_token(
    name: str,
    description: str,
    ticker: str,
    image: str,
):
    """
    Launches a new token on the Swarms platform via the Launchpad API.

    This function sends a POST request to the Swarms API to create a new token with the specified parameters.
    It uses the API key and the configured private key for authentication and authorization.

    Args:
        name (str): The name of the token to be launched (e.g., "My Cool Token").
        description (str): A brief description of the token's purpose or use-case.
        ticker (str): The ticker symbol for the token (e.g., "MCT").
        image (str): A URL or base64-encoded string representing the token image/logo.

    Returns:
        dict: The parsed JSON response from the API containing token creation details
            or error information. Expected successful response keys may include:
            - "success" (bool): Whether the token was created successfully.
            - "token_id" (str or int): The ID of the created token.
            - "message" (str): Additional info or status messages.

    Raises:
        httpx.RequestError: If there is a network problem or the API is unreachable.
        httpx.HTTPStatusError: If the server returns an error status code.
        (Note: these errors will not be caught here; callers should handle as needed.)

    Example:
        >>> result = launch_token(
        ...     name="Test Token",
        ...     description="Token for demo purposes.",
        ...     ticker="TT",
        ...     image="https://example.com/img.png"
        ... )
        >>> print(result["success"])
        True

    Security Notes:
        - The `PRIVATE_KEY` is sent as part of the payload. Keep your keys secure and
          be careful not to expose them.
        - Ensure that SWARMS_API_KEY and PRIVATE_KEY are set in your environment variables.

    """
    url = f"{BASE_URL}/api/token/launch"
    headers = {
        "Authorization": f"Bearer {SWARMS_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "name": name,
        "description": description,
        "ticker": ticker,
        "image": image,
        "private_key": PRIVATE_KEY,
    }
    response = httpx.post(url, headers=headers, json=data)
    output = response.json()

    return json.dumps(output, indent=4)


def claim_fees_httpx(
    contract_address: Optional[str] = None,
) -> str:
    """
    Claims fees from the Swarms API using httpx.

    Args:
        contract_address (Optional[str]): The contract address ("ca") to claim from.
            Defaults to a preset address if not provided.
        private_key (Optional[str]): The base58 private key for authorization.
            Must be provided by the caller for security.

    Returns:
        dict: The parsed JSON response from the API containing signature, amount claimed,
            fees, or error information.
    """
    url = "https://swarms.world/api/product/claimfees"
    private_key = PRIVATE_KEY

    payload = {"ca": contract_address, "privateKey": private_key}

    try:
        response = httpx.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        return json.dumps(data, indent=4)
    except httpx.HTTPStatusError as exc:
        # If error response is JSON, attempt to extract "error" or fall back to content string
        try:
            error_data = exc.response.json()
            print(
                "Error:", error_data.get("error", exc.response.text)
            )
            return error_data
        except Exception:
            print("Error:", exc.response.text)
            return {"error": exc.response.text}
    except httpx.RequestError as exc:
        print(
            f"Network error while requesting {exc.request.url!r}: {exc}"
        )
        return {"error": str(exc)}


# Initialize the agent
agent = Agent(
    agent_name="Meme-Coin-Launch-Pro",
    agent_description="Expert agent for launching and branding meme coins on the Swarms launchpad; specializes in naming, tickers, viral copy, and token creation.",
    system_prompt=MEME_COIN_LAUNCH_SYSTEM_PROMPT,
    model_name="anthropic/claude-sonnet-4-5",
    dynamic_temperature_enabled=True,
    max_loops=1,
    dynamic_context_window=True,
    streaming_on=True,
    interactive=False,
    top_p=None,
    tools=[claim_fees_httpx, launch_token],
)

out = agent.run(
    task="I want to launch a meme coin about a cat that thinks it's a CEO. Suggest a name, ticker, short description, and what kind of image would work. If I say 'launch it', use the launch_token tool with those details (you can use a placeholder image URL if I don't provide one).",
)

print(out)
