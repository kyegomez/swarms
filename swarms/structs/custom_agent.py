import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import httpx
from loguru import logger


@dataclass
class AgentResponse:
    """Data class to hold agent response information"""

    status_code: int
    content: str
    headers: Dict[str, str]
    json_data: Optional[Dict[str, Any]] = None
    success: bool = False
    error_message: Optional[str] = None


class CustomAgent:
    """
    A custom HTTP agent class for making POST requests using httpx.

    Features:
    - Configurable headers and payload
    - Both sync and async execution
    - Built-in error handling and logging
    - Flexible response handling
    - Name and description
    """

    def __init__(
        self,
        name: str,
        description: str,
        base_url: str,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        verify_ssl: bool = True,
        *args,
        **kwargs,
    ):
        """
        Initialize the Custom Agent.

        Args:
            base_url: Base URL for the API endpoint
            endpoint: API endpoint path
            headers: Default headers to include in requests
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
        """
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint.lstrip("/")
        self.default_headers = headers or {}
        self.timeout = timeout
        self.verify_ssl = verify_ssl

        # Default headers
        if "Content-Type" not in self.default_headers:
            self.default_headers["Content-Type"] = "application/json"

        logger.info(
            f"CustomAgent initialized for {self.base_url}/{self.endpoint}"
        )

    def _prepare_headers(
        self, additional_headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """Merge default headers with additional headers."""
        headers = self.default_headers.copy()
        if additional_headers:
            headers.update(additional_headers)
        return headers

    def _prepare_payload(
        self, payload: Union[Dict, str, bytes]
    ) -> Union[str, bytes]:
        """Prepare the payload for the request."""
        if isinstance(payload, dict):
            return json.dumps(payload)
        return payload

    def _parse_response(
        self, response: httpx.Response
    ) -> AgentResponse:
        """Parse httpx response into AgentResponse object."""
        try:
            # Try to parse JSON if possible
            json_data = None
            if response.headers.get("content-type", "").startswith(
                "application/json"
            ):
                try:
                    json_data = response.json()
                except json.JSONDecodeError:
                    pass

            return AgentResponse(
                status_code=response.status_code,
                content=response.text,
                headers=dict(response.headers),
                json_data=json_data,
                success=200 <= response.status_code < 300,
                error_message=(
                    None
                    if 200 <= response.status_code < 300
                    else f"HTTP {response.status_code}"
                ),
            )
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return AgentResponse(
                status_code=response.status_code,
                content=response.text,
                headers=dict(response.headers),
                success=False,
                error_message=str(e),
            )

    def _extract_content(self, response_data: Dict[str, Any]) -> str:
        """
        Extract message content from API response, supporting multiple formats.

        Args:
            response_data: Parsed JSON response from API

        Returns:
            str: Extracted message content
        """
        try:
            # OpenAI format
            if (
                "choices" in response_data
                and response_data["choices"]
            ):
                choice = response_data["choices"][0]
                if (
                    "message" in choice
                    and "content" in choice["message"]
                ):
                    return choice["message"]["content"]
                elif "text" in choice:
                    return choice["text"]

            # Anthropic format
            elif (
                "content" in response_data
                and response_data["content"]
            ):
                if isinstance(response_data["content"], list):
                    # Extract text from content blocks
                    text_parts = []
                    for content_block in response_data["content"]:
                        if (
                            isinstance(content_block, dict)
                            and "text" in content_block
                        ):
                            text_parts.append(content_block["text"])
                        elif isinstance(content_block, str):
                            text_parts.append(content_block)
                    return "".join(text_parts)
                elif isinstance(response_data["content"], str):
                    return response_data["content"]

            # Generic fallback - look for common content fields
            elif "text" in response_data:
                return response_data["text"]
            elif "message" in response_data:
                return response_data["message"]
            elif "response" in response_data:
                return response_data["response"]

            # If no known format, return the entire response as JSON string
            logger.warning(
                "Unknown response format, returning full response"
            )
            return json.dumps(response_data, indent=2)

        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            return json.dumps(response_data, indent=2)

    def run(
        self,
        payload: Union[Dict[str, Any], str, bytes],
        additional_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> str:
        """
        Execute a synchronous POST request.

        Args:
            payload: Request body/payload
            additional_headers: Additional headers for this request
            **kwargs: Additional httpx client options

        Returns:
            str: Extracted message content from response
        """
        url = f"{self.base_url}/{self.endpoint}"
        request_headers = self._prepare_headers(additional_headers)
        request_payload = self._prepare_payload(payload)

        logger.info(f"Making POST request to: {url}")

        try:
            with httpx.Client(
                timeout=self.timeout, verify=self.verify_ssl, **kwargs
            ) as client:
                response = client.post(
                    url,
                    content=request_payload,
                    headers=request_headers,
                )

                if 200 <= response.status_code < 300:
                    logger.info(
                        f"Request successful: {response.status_code}"
                    )
                    try:
                        response_data = response.json()
                        return self._extract_content(response_data)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Response is not JSON, returning raw text"
                        )
                        return response.text
                else:
                    logger.warning(
                        f"Request failed: {response.status_code}"
                    )
                    return f"Error: HTTP {response.status_code} - {response.text}"

        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            return f"Request error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return f"Unexpected error: {str(e)}"

    async def run_async(
        self,
        payload: Union[Dict[str, Any], str, bytes],
        additional_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> str:
        """
        Execute an asynchronous POST request.

        Args:
            payload: Request body/payload
            additional_headers: Additional headers for this request
            **kwargs: Additional httpx client options

        Returns:
            str: Extracted message content from response
        """
        url = f"{self.base_url}/{self.endpoint}"
        request_headers = self._prepare_headers(additional_headers)
        request_payload = self._prepare_payload(payload)

        logger.info(f"Making async POST request to: {url}")

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, verify=self.verify_ssl, **kwargs
            ) as client:
                response = await client.post(
                    url,
                    content=request_payload,
                    headers=request_headers,
                )

                if 200 <= response.status_code < 300:
                    logger.info(
                        f"Async request successful: {response.status_code}"
                    )
                    try:
                        response_data = response.json()
                        return self._extract_content(response_data)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Async response is not JSON, returning raw text"
                        )
                        return response.text
                else:
                    logger.warning(
                        f"Async request failed: {response.status_code}"
                    )
                    return f"Error: HTTP {response.status_code} - {response.text}"

        except httpx.RequestError as e:
            logger.error(f"Async request error: {e}")
            return f"Request error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected async error: {e}")
            return f"Unexpected error: {str(e)}"


# # Example usage with Anthropic API
# if __name__ == "__main__":
#     # Initialize the agent for Anthropic API
#     anthropic_agent = CustomAgent(
#         base_url="https://api.anthropic.com",
#         endpoint="v1/messages",
#         headers={
#             "x-api-key": "your-anthropic-api-key-here",
#             "anthropic-version": "2023-06-01"
#         }
#     )

#     # Example payload for Anthropic API
#     payload = {
#         "model": "claude-3-sonnet-20240229",
#         "max_tokens": 1000,
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "Hello! Can you explain what artificial intelligence is?"
#             }
#         ]
#     }

#     # Make the request
#     try:
#         response = anthropic_agent.run(payload)
#         print("Anthropic API Response:")
#         print(response)
#     except Exception as e:
#         print(f"Error: {e}")

#     # Example with async usage
#     # import asyncio
#     #
#     # async def async_example():
#     #     response = await anthropic_agent.run_async(payload)
#     #     print("Async Anthropic API Response:")
#     #     print(response)
#     #
#     # Uncomment to run async example
#     # asyncio.run(async_example())
