import asyncio
import os
from typing import List, Optional

import httpx
from loguru import logger
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from swarms.structs.agent import agent_roles
from swarms.structs.swarm_router import SwarmType


class Agent(BaseModel):
    agent_name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    model_name: Optional[str] = None
    role: Optional[agent_roles] = "worker"
    max_loops: Optional[int] = Field(default=1, ge=1)


class SwarmRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    agents: Optional[List[Agent]] = None
    max_loops: Optional[int] = Field(default=1, ge=1)
    swarm_type: Optional[SwarmType] = "ConcurrentWorkflow"
    task: Optional[str] = None


class SwarmResponse(BaseModel):
    swarm_id: Optional[str] = None
    status: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: Optional[str] = None
    version: Optional[str] = None  # Make version optional


class SwarmAPIError(Exception):
    """Base exception for Swarms API errors."""

    pass


class SwarmAuthenticationError(SwarmAPIError):
    """Raised when authentication fails."""

    pass


class SwarmValidationError(SwarmAPIError):
    """Raised when request validation fails."""

    pass


class SwarmClient:
    """Production-grade client for the Swarms API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://swarms-api-285321057562.us-east1.run.app",
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """Initialize the Swarms API client.

        Args:
            api_key: API key for authentication. If not provided, looks for SWARMS_API_KEY env var
            base_url: Base URL for the API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.api_key = api_key or os.getenv("SWARMS_API_KEY")

        if not self.api_key:
            raise SwarmAuthenticationError(
                "API key not provided and SWARMS_API_KEY env var not found"
            )

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

        # Configure logging
        logger.add(
            "swarms_client.log",
            rotation="100 MB",
            retention="1 week",
            level="INFO",
        )

        # Setup HTTP client
        self.client = httpx.Client(
            timeout=timeout,
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
            },
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def health_check(self) -> HealthResponse:
        """Check the API health status."""
        try:
            response = self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return HealthResponse(**response.json())
        except httpx.HTTPError as e:
            logger.error(f"Health check failed: {str(e)}")
            raise SwarmAPIError(f"Health check failed: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def create_swarm(
        self, swarm_request: SwarmRequest
    ) -> SwarmResponse:
        """Create and run a new swarm.

        Args:
            swarm_request: SwarmRequest object containing the swarm configuration

        Returns:
            SwarmResponse object with the results
        """
        try:
            response = self.client.post(
                f"{self.base_url}/v1/swarm/completions",
                json=swarm_request.model_dump(),
            )
            response.raise_for_status()
            return SwarmResponse(**response.json())
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise SwarmAuthenticationError("Invalid API key")
            elif e.response.status_code == 422:
                raise SwarmValidationError(
                    "Invalid request parameters"
                )
            logger.error(f"Swarm creation failed: {str(e)}")
            raise SwarmAPIError(f"Swarm creation failed: {str(e)}")
        except Exception as e:
            logger.error(
                f"Unexpected error during swarm creation: {str(e)}"
            )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def create_batch_swarms(
        self, swarm_requests: List[SwarmRequest]
    ) -> List[SwarmResponse]:
        """Create and run multiple swarms in batch.

        Args:
            swarm_requests: List of SwarmRequest objects

        Returns:
            List of SwarmResponse objects
        """
        try:
            response = self.client.post(
                f"{self.base_url}/v1/swarm/batch/completions",
                json=[req.model_dump() for req in swarm_requests],
            )
            response.raise_for_status()
            return [SwarmResponse(**resp) for resp in response.json()]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise SwarmAuthenticationError("Invalid API key")
            elif e.response.status_code == 422:
                raise SwarmValidationError(
                    "Invalid request parameters"
                )
            logger.error(f"Batch swarm creation failed: {str(e)}")
            raise SwarmAPIError(
                f"Batch swarm creation failed: {str(e)}"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error during batch swarm creation: {str(e)}"
            )
            raise

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Example usage

async def main():
    async with SwarmClient(api_key=os.getenv("SWARMS_API_KEY")) as client:
        health = await client.health_check()
        print(health)

if __name__ == "__main__":
    asyncio.run(main())
