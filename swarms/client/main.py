import json
import os
from typing import List, Literal, Optional

import httpx
from swarms.utils.loguru_logger import initialize_logger
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from swarms.structs.swarm_router import SwarmType
from typing import Any

logger = initialize_logger(log_folder="swarms_api")


class AgentInput(BaseModel):
    agent_name: Optional[str] = Field(
        None,
        description="The name of the agent, limited to 100 characters.",
        max_length=100,
    )
    description: Optional[str] = Field(
        None,
        description="A detailed description of the agent's purpose and capabilities, up to 500 characters.",
        max_length=500,
    )
    system_prompt: Optional[str] = Field(
        None,
        description="The initial prompt or instructions given to the agent.",
    )
    model_name: Optional[str] = Field(
        "gpt-4o",
        description="The name of the model used by the agent. Model names can be configured like provider/model_name",
    )
    auto_generate_prompt: Optional[bool] = Field(
        False,
        description="Indicates whether the agent should automatically generate prompts.",
    )
    max_tokens: Optional[int] = Field(
        8192,
        description="The maximum number of tokens the agent can use in its responses.",
    )
    temperature: Optional[float] = Field(
        0.5,
        description="Controls the randomness of the agent's responses; higher values result in more random outputs.",
    )
    role: Optional[str] = Field(
        "worker",
        description="The role assigned to the agent, such as 'worker' or 'manager'.",
    )
    max_loops: Optional[int] = Field(
        1,
        description="The maximum number of iterations the agent is allowed to perform.",
    )
    dynamic_temperature_enabled: Optional[bool] = Field(
        True,
        description="Indicates whether the agent should use dynamic temperature.",
    )


class SwarmRequest(BaseModel):
    name: Optional[str] = Field(
        "swarms-01",
        description="The name of the swarm, limited to 100 characters.",
        max_length=100,
    )
    description: Optional[str] = Field(
        None,
        description="A comprehensive description of the swarm's objectives and scope, up to 500 characters.",
        max_length=500,
    )
    agents: Optional[List[AgentInput]] = Field(
        None,
        description="A list of agents that are part of the swarm.",
    )
    max_loops: Optional[int] = Field(
        1,
        description="The maximum number of iterations the swarm can execute.",
    )
    swarm_type: Optional[SwarmType] = Field(
        None,
        description="The type of swarm, defining its operational structure and behavior.",
    )
    rearrange_flow: Optional[str] = Field(
        None,
        description="The flow or sequence in which agents are rearranged during the swarm's operation.",
    )
    task: Optional[str] = Field(
        None,
        description="The specific task or objective the swarm is designed to accomplish.",
    )
    img: Optional[str] = Field(
        None,
        description="A URL to an image associated with the swarm, if applicable.",
    )
    return_history: Optional[bool] = Field(
        True,
        description="Determines whether the full history of the swarm's operations should be returned.",
    )
    rules: Optional[str] = Field(
        None,
        description="Any specific rules or guidelines that the swarm should follow.",
    )
    output_type: Optional[str] = Field(
        "str",
        description="The format in which the swarm's output should be returned, such as 'str', 'json', or 'dict'.",
    )


# class SwarmResponse(BaseModel):
#     swarm_id: str
#     status: str
#     result: Optional[str]
#     error: Optional[str]


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
        base_url: str = "https://api.swarms.world",
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
            logger.error(
                "API key not provided and SWARMS_API_KEY env var not found"
            )
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
        logger.info(
            "SwarmsAPIClient initialized with base_url: {}",
            self.base_url,
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
        logger.info("Performing health check")
        try:
            response = self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
            health_response = HealthResponse(**response.json())
            logger.info("Health check successful")
            return self.format_output(
                health_response, self.format_type
            )
        except httpx.HTTPError as e:
            logger.error("Health check failed: {}", str(e))
            raise SwarmAPIError(f"Health check failed: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def arun(self, swarm_request: SwarmRequest) -> Any:
        """Create and run a new swarm.

        Args:
            swarm_request: SwarmRequest object containing the swarm configuration
            output_format: Desired output format ('pydantic', 'json', 'dict')

        Returns:
            SwarmResponse object or formatted output
        """
        logger.info(
            "Creating and running a new swarm with request: {}",
            swarm_request,
        )
        try:
            response = self.client.post(
                f"{self.base_url}/v1/swarm/completions",
                json=swarm_request.model_dump(),
            )
            response.raise_for_status()
            logger.info("Swarm creation and run successful")
            return self.format_output(
                response.json(), self.format_type
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.error("Invalid API key")
                raise SwarmAuthenticationError("Invalid API key")
            elif e.response.status_code == 422:
                logger.error("Invalid request parameters")
                raise SwarmValidationError(
                    "Invalid request parameters"
                )
            logger.error("Swarm creation failed: {}", str(e))
            raise SwarmAPIError(f"Swarm creation failed: {str(e)}")
        except Exception as e:
            logger.error(
                "Unexpected error during swarm creation: {}", str(e)
            )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def run(self, swarm_request: SwarmRequest) -> Any:
        """Create and run a new swarm.

        Args:
            swarm_request: SwarmRequest object containing the swarm configuration
            output_format: Desired output format ('pydantic', 'json', 'dict')

        Returns:
            SwarmResponse object or formatted output
        """
        logger.info(
            "Creating and running a new swarm with request: {}",
            swarm_request,
        )
        try:
            response = self.client.post(
                f"{self.base_url}/v1/swarm/completions",
                json=swarm_request.model_dump(),
            )
            print(response.json())
            logger.info("Swarm creation and run successful")
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.error("Invalid API key")
                raise SwarmAuthenticationError("Invalid API key")
            elif e.response.status_code == 422:
                logger.error("Invalid request parameters")
                raise SwarmValidationError(
                    "Invalid request parameters"
                )
            logger.error("Swarm creation failed: {}", str(e))
            raise SwarmAPIError(f"Swarm creation failed: {str(e)}")
        except Exception as e:
            logger.error(
                "Unexpected error during swarm creation: {}", str(e)
            )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def run_batch(
        self, swarm_requests: List[SwarmRequest]
    ) -> List[Any]:
        """Create and run multiple swarms in batch.

        Args:
            swarm_requests: List of SwarmRequest objects
            output_format: Desired output format ('pydantic', 'json', 'dict')

        Returns:
            List of SwarmResponse objects or formatted outputs
        """
        logger.info(
            "Creating and running batch swarms with requests: {}",
            swarm_requests,
        )
        try:
            response = self.client.post(
                f"{self.base_url}/v1/swarm/batch/completions",
                json=[req.model_dump() for req in swarm_requests],
            )
            response.raise_for_status()
            logger.info("Batch swarm creation and run successful")
            return [
                self.format_output(resp, self.format_type)
                for resp in response.json()
            ]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.error("Invalid API key")
                raise SwarmAuthenticationError("Invalid API key")
            elif e.response.status_code == 422:
                logger.error("Invalid request parameters")
                raise SwarmValidationError(
                    "Invalid request parameters"
                )
            logger.error("Batch swarm creation failed: {}", str(e))
            raise SwarmAPIError(
                f"Batch swarm creation failed: {str(e)}"
            )
        except Exception as e:
            logger.error(
                "Unexpected error during batch swarm creation: {}",
                str(e),
            )
            raise

    def get_logs(self):
        logger.info("Retrieving logs")
        try:
            response = self.client.get(
                f"{self.base_url}/v1/swarm/logs"
            )
            response.raise_for_status()
            logs = response.json()
            logger.info("Logs retrieved successfully")
            return self.format_output(logs, self.format_type)
        except httpx.HTTPError as e:
            logger.error("Failed to retrieve logs: {}", str(e))
            raise SwarmAPIError(f"Failed to retrieve logs: {str(e)}")

    def format_output(self, data, output_format: str):
        """Format the output based on the specified format.

        Args:
            data: The data to format
            output_format: The desired output format ('pydantic', 'json', 'dict')

        Returns:
            Formatted data
        """
        logger.info(
            "Formatting output with format: {}", output_format
        )
        if output_format == "json":
            return (
                data.model_dump_json(indent=4)
                if isinstance(data, BaseModel)
                else json.dumps(data)
            )
        elif output_format == "dict":
            return (
                data.model_dump()
                if isinstance(data, BaseModel)
                else data
            )
        return data  # Default to returning the pydantic model

    def close(self):
        """Close the HTTP client."""
        logger.info("Closing HTTP client")
        self.client.close()

    async def __aenter__(self):
        logger.info("Entering async context")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logger.info("Exiting async context")
        self.close()
