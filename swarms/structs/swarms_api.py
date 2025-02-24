import json
import os
from typing import List, Literal, Optional

import httpx
from swarms.utils.loguru_logger import initialize_logger
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from swarms.structs.swarm_router import SwarmType

logger = initialize_logger(log_folder="swarms_api")


class AgentInput(BaseModel):
    agent_name: Optional[str] = Field(None, description="Agent Name", max_length=100)
    description: Optional[str] = Field(None, description="Description", max_length=500)
    system_prompt: Optional[str] = Field(
        None, description="System Prompt", max_length=500
    )
    model_name: Optional[str] = Field(
        "gpt-4o", description="Model Name", max_length=500
    )
    auto_generate_prompt: Optional[bool] = Field(
        False, description="Auto Generate Prompt"
    )
    max_tokens: Optional[int] = Field(None, description="Max Tokens")
    temperature: Optional[float] = Field(0.5, description="Temperature")
    role: Optional[str] = Field("worker", description="Role")
    max_loops: Optional[int] = Field(1, description="Max Loops")


class SwarmRequest(BaseModel):
    name: Optional[str] = Field(None, description="Swarm Name", max_length=100)
    description: Optional[str] = Field(None, description="Description", max_length=500)
    agents: Optional[List[AgentInput]] = Field(None, description="Agents")
    max_loops: Optional[int] = Field(None, description="Max Loops")
    swarm_type: Optional[SwarmType] = Field(None, description="Swarm Type")
    rearrange_flow: Optional[str] = Field(None, description="Flow")
    task: Optional[str] = Field(None, description="Task")
    img: Optional[str] = Field(None, description="Img")
    return_history: Optional[bool] = Field(True, description="Return History")
    rules: Optional[str] = Field(None, description="Rules")

class SwarmResponse(BaseModel):
    swarm_id: str
    status: str
    result: Optional[str]
    error: Optional[str]


class HealthResponse(BaseModel):
    status: str
    version: str


class SwarmAPIError(Exception):
    """Base exception for Swarms API errors."""

    pass


class SwarmAuthenticationError(SwarmAPIError):
    """Raised when authentication fails."""

    pass


class SwarmValidationError(SwarmAPIError):
    """Raised when request validation fails."""

    pass


class SwarmsAPIClient:
    """Production-grade client for the Swarms API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://swarms-api-285321057562.us-east1.run.app",
        timeout: int = 30,
        max_retries: int = 3,
        format_type: Literal["pydantic", "json", "dict"] = "pydantic",
    ):
        """Initialize the Swarms API client.

        Args:
            api_key: API key for authentication. If not provided, looks for SWARMS_API_KEY env var
            base_url: Base URL for the API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            format_type: Desired output format ('pydantic', 'json', 'dict')
        """
        self.api_key = api_key or os.getenv("SWARMS_API_KEY")

        if not self.api_key:
            raise SwarmAuthenticationError(
                "API key not provided and SWARMS_API_KEY env var not found"
            )

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.format_type = format_type
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
        """Check the API health status.

        Args:
            output_format: Desired output format ('pydantic', 'json', 'dict')

        Returns:
            HealthResponse object or formatted output
        """
        try:
            response = self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
            health_response = HealthResponse(**response.json())
            return self.format_output(health_response, self.format_type)
        except httpx.HTTPError as e:
            logger.error(f"Health check failed: {str(e)}")
            raise SwarmAPIError(f"Health check failed: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def run(
        self, swarm_request: SwarmRequest
    ) -> SwarmResponse:
        """Create and run a new swarm.

        Args:
            swarm_request: SwarmRequest object containing the swarm configuration
            output_format: Desired output format ('pydantic', 'json', 'dict')

        Returns:
            SwarmResponse object or formatted output
        """
        try:
            response = self.client.post(
                f"{self.base_url}/v1/swarm/completions",
                json=swarm_request.model_dump(),
            )
            response.raise_for_status()
            swarm_response = SwarmResponse(**response.json())
            return self.format_output(swarm_response, self.format_type)
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
    async def run_batch(
        self, swarm_requests: List[SwarmRequest]
    ) -> List[SwarmResponse]:
        """Create and run multiple swarms in batch.

        Args:
            swarm_requests: List of SwarmRequest objects
            output_format: Desired output format ('pydantic', 'json', 'dict')

        Returns:
            List of SwarmResponse objects or formatted outputs
        """
        try:
            response = self.client.post(
                f"{self.base_url}/v1/swarm/batch/completions",
                json=[req.model_dump() for req in swarm_requests],
            )
            response.raise_for_status()
            swarm_responses = [SwarmResponse(**resp) for resp in response.json()]
            return [self.format_output(resp, self.format_type) for resp in swarm_responses]
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
        
    def get_logs(self):
        response = self.client.get(f"{self.base_url}/v1/swarm/logs")
        response.raise_for_status()
        logs = response.json()
        return self.format_output(logs, self.format_type)

    def format_output(self, data, output_format: str):
        """Format the output based on the specified format.

        Args:
            data: The data to format
            output_format: The desired output format ('pydantic', 'json', 'dict')

        Returns:
            Formatted data
        """
        if output_format == "json":
            return data.model_dump_json(indent=4) if isinstance(data, BaseModel) else json.dumps(data)
        elif output_format == "dict":
            return data.model_dump() if isinstance(data, BaseModel) else data
        return data  # Default to returning the pydantic model

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()
