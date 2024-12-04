import os
from fastapi import (
    FastAPI,
    HTTPException,
    status,
    Query,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from loguru import logger
import uvicorn
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import traceback

from swarms import Agent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Loguru
logger.add(
    "logs/api_{time}.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO",
    format="{time} {level} {message}",
    backtrace=True,
    diagnose=True,
)


class AgentStatus(str, Enum):
    """Enum for agent status."""

    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class AgentConfig(BaseModel):
    """Configuration model for creating a new agent."""

    agent_name: str = Field(..., description="Name of the agent")
    model_name: str = Field(
        ...,
        description="Name of the llm you want to use provided by litellm",
    )
    description: str = Field(
        default="", description="Description of the agent's purpose"
    )
    system_prompt: str = Field(
        ..., description="System prompt for the agent"
    )
    model_name: str = Field(
        default="gpt-4", description="Model name to use"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for the model",
    )
    max_loops: int = Field(
        default=1, ge=1, description="Maximum number of loops"
    )
    autosave: bool = Field(
        default=True, description="Enable autosave"
    )
    dashboard: bool = Field(
        default=False, description="Enable dashboard"
    )
    verbose: bool = Field(
        default=True, description="Enable verbose output"
    )
    dynamic_temperature_enabled: bool = Field(
        default=True, description="Enable dynamic temperature"
    )
    user_name: str = Field(
        default="default_user", description="Username for the agent"
    )
    retry_attempts: int = Field(
        default=1, ge=1, description="Number of retry attempts"
    )
    context_length: int = Field(
        default=200000, ge=1000, description="Context length"
    )
    output_type: str = Field(
        default="string", description="Output type (string or json)"
    )
    streaming_on: bool = Field(
        default=False, description="Enable streaming"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorizing the agent",
    )


class AgentUpdate(BaseModel):
    """Model for updating agent configuration."""

    description: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_loops: Optional[int] = None
    tags: Optional[List[str]] = None
    status: Optional[AgentStatus] = None


class AgentSummary(BaseModel):
    """Summary model for agent listing."""

    agent_id: UUID
    agent_name: str
    description: str
    created_at: datetime
    last_used: datetime
    total_completions: int
    tags: List[str]
    status: AgentStatus


class AgentMetrics(BaseModel):
    """Model for agent performance metrics."""

    total_completions: int
    average_response_time: float
    error_rate: float
    last_24h_completions: int
    total_tokens_used: int
    uptime_percentage: float
    success_rate: float
    peak_tokens_per_minute: int


class CompletionRequest(BaseModel):
    """Model for completion requests."""

    prompt: str = Field(..., description="The prompt to process")
    agent_id: UUID = Field(..., description="ID of the agent to use")
    max_tokens: Optional[int] = Field(
        None, description="Maximum tokens to generate"
    )
    temperature_override: Optional[float] = None
    stream: bool = Field(
        default=False, description="Enable streaming response"
    )


class CompletionResponse(BaseModel):
    """Model for completion responses."""

    agent_id: UUID
    response: str
    metadata: Dict[str, Any]
    timestamp: datetime
    processing_time: float
    token_usage: Dict[str, int]


class AgentStore:
    """Enhanced store for managing agents."""

    def __init__(self):
        self.agents: Dict[UUID, Agent] = {}
        self.agent_metadata: Dict[UUID, Dict[str, Any]] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure required directories exist."""
        Path("logs").mkdir(exist_ok=True)
        Path("states").mkdir(exist_ok=True)

    async def create_agent(self, config: AgentConfig) -> UUID:
        """Create a new agent with the given configuration."""
        try:

            agent = Agent(
                agent_name=config.agent_name,
                system_prompt=config.system_prompt,
                model_name=config.model_name,
                max_loops=config.max_loops,
                autosave=config.autosave,
                dashboard=config.dashboard,
                verbose=config.verbose,
                dynamic_temperature_enabled=config.dynamic_temperature_enabled,
                saved_state_path=f"states/{config.agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                user_name=config.user_name,
                retry_attempts=config.retry_attempts,
                context_length=config.context_length,
                return_step_meta=True,
                output_type="str",
                streaming_on=config.streaming_on,
            )

            agent_id = uuid4()
            self.agents[agent_id] = agent
            self.agent_metadata[agent_id] = {
                "description": config.description,
                "created_at": datetime.utcnow(),
                "last_used": datetime.utcnow(),
                "total_completions": 0,
                "tags": config.tags,
                "total_tokens": 0,
                "error_count": 0,
                "response_times": [],
                "status": AgentStatus.IDLE,
                "start_time": datetime.utcnow(),
                "downtime": timedelta(),
                "successful_completions": 0,
            }

            logger.info(f"Created agent with ID: {agent_id}")
            return agent_id

        except Exception as e:
            logger.error(f"Error creating agent: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create agent: {str(e)}",
            )

    async def get_agent(self, agent_id: UUID) -> Agent:
        """Retrieve an agent by ID."""
        agent = self.agents.get(agent_id)
        if not agent:
            logger.error(f"Agent not found: {agent_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found",
            )
        return agent

    async def update_agent(
        self, agent_id: UUID, update: AgentUpdate
    ) -> None:
        """Update agent configuration."""
        agent = await self.get_agent(agent_id)
        metadata = self.agent_metadata[agent_id]

        if update.system_prompt:
            agent.system_prompt = update.system_prompt
        if update.temperature is not None:
            agent.llm.temperature = update.temperature
        if update.max_loops is not None:
            agent.max_loops = update.max_loops
        if update.tags is not None:
            metadata["tags"] = update.tags
        if update.description is not None:
            metadata["description"] = update.description
        if update.status is not None:
            metadata["status"] = update.status
            if update.status == AgentStatus.MAINTENANCE:
                metadata["downtime"] += (
                    datetime.utcnow() - metadata["last_used"]
                )

        logger.info(f"Updated agent {agent_id}")

    async def list_agents(
        self,
        tags: Optional[List[str]] = None,
        status: Optional[AgentStatus] = None,
    ) -> List[AgentSummary]:
        """List all agents, optionally filtered by tags and status."""
        summaries = []
        for agent_id, agent in self.agents.items():
            metadata = self.agent_metadata[agent_id]

            # Apply filters
            if tags and not any(
                tag in metadata["tags"] for tag in tags
            ):
                continue
            if status and metadata["status"] != status:
                continue

            summaries.append(
                AgentSummary(
                    agent_id=agent_id,
                    agent_name=agent.agent_name,
                    description=metadata["description"],
                    created_at=metadata["created_at"],
                    last_used=metadata["last_used"],
                    total_completions=metadata["total_completions"],
                    tags=metadata["tags"],
                    status=metadata["status"],
                )
            )
        return summaries

    async def get_agent_metrics(self, agent_id: UUID) -> AgentMetrics:
        """Get performance metrics for an agent."""
        metadata = self.agent_metadata[agent_id]
        response_times = metadata["response_times"]

        # Calculate metrics
        total_time = datetime.utcnow() - metadata["start_time"]
        uptime = total_time - metadata["downtime"]
        uptime_percentage = (
            uptime.total_seconds() / total_time.total_seconds()
        ) * 100

        success_rate = (
            metadata["successful_completions"]
            / metadata["total_completions"]
            * 100
            if metadata["total_completions"] > 0
            else 0
        )

        return AgentMetrics(
            total_completions=metadata["total_completions"],
            average_response_time=(
                sum(response_times) / len(response_times)
                if response_times
                else 0
            ),
            error_rate=(
                metadata["error_count"]
                / metadata["total_completions"]
                if metadata["total_completions"] > 0
                else 0
            ),
            last_24h_completions=sum(
                1
                for t in response_times
                if (datetime.utcnow() - t).days < 1
            ),
            total_tokens_used=metadata["total_tokens"],
            uptime_percentage=uptime_percentage,
            success_rate=success_rate,
            peak_tokens_per_minute=max(
                metadata.get("tokens_per_minute", [0])
            ),
        )

    async def clone_agent(
        self, agent_id: UUID, new_name: str
    ) -> UUID:
        """Clone an existing agent with a new name."""
        original_agent = await self.get_agent(agent_id)
        original_metadata = self.agent_metadata[agent_id]

        config = AgentConfig(
            agent_name=new_name,
            description=f"Clone of {original_agent.agent_name}",
            system_prompt=original_agent.system_prompt,
            model_name=original_agent.llm.model_name,
            temperature=original_agent.llm.temperature,
            max_loops=original_agent.max_loops,
            tags=original_metadata["tags"],
        )

        return await self.create_agent(config)

    async def delete_agent(self, agent_id: UUID) -> None:
        """Delete an agent."""
        if agent_id not in self.agents:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found",
            )

        # Clean up any resources
        agent = self.agents[agent_id]
        if agent.autosave and os.path.exists(agent.saved_state_path):
            os.remove(agent.saved_state_path)

        del self.agents[agent_id]
        del self.agent_metadata[agent_id]
        logger.info(f"Deleted agent {agent_id}")

    async def process_completion(
        self,
        agent: Agent,
        prompt: str,
        agent_id: UUID,
        max_tokens: Optional[int] = None,
        temperature_override: Optional[float] = None,
    ) -> CompletionResponse:
        """Process a completion request using the specified agent."""
        start_time = datetime.utcnow()
        metadata = self.agent_metadata[agent_id]

        try:
            # Update agent status
            metadata["status"] = AgentStatus.PROCESSING
            metadata["last_used"] = start_time

            # Apply temporary overrides if specified
            original_temp = agent.llm.temperature
            if temperature_override is not None:
                agent.llm.temperature = temperature_override

            # Process the completion
            response = agent.run(prompt)

            # Reset overrides
            if temperature_override is not None:
                agent.llm.temperature = original_temp

            # Update metrics
            processing_time = (
                datetime.utcnow() - start_time
            ).total_seconds()
            metadata["response_times"].append(processing_time)
            metadata["total_completions"] += 1
            metadata["successful_completions"] += 1

            # Estimate token usage (this is a rough estimate)
            prompt_tokens = len(prompt.split()) * 1.3
            completion_tokens = len(response.split()) * 1.3
            total_tokens = int(prompt_tokens + completion_tokens)
            metadata["total_tokens"] += total_tokens

            # Update tokens per minute tracking
            current_minute = datetime.utcnow().replace(
                second=0, microsecond=0
            )
            if "tokens_per_minute" not in metadata:
                metadata["tokens_per_minute"] = {}
            metadata["tokens_per_minute"][current_minute] = (
                metadata["tokens_per_minute"].get(current_minute, 0)
                + total_tokens
            )

            return CompletionResponse(
                agent_id=agent_id,
                response=response,
                metadata={
                    "agent_name": agent.agent_name,
                    "model_name": agent.llm.model_name,
                    "temperature": agent.llm.temperature,
                },
                timestamp=datetime.utcnow(),
                processing_time=processing_time,
                token_usage={
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                    "total_tokens": total_tokens,
                },
            )

        except Exception as e:
            metadata["error_count"] += 1
            metadata["status"] = AgentStatus.ERROR
            logger.error(
                f"Error in completion processing: {str(e)}\n{traceback.format_exc()}"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing completion: {str(e)}",
            )
        finally:
            metadata["status"] = AgentStatus.IDLE


class SwarmsAPI:
    """Enhanced API class for Swarms agent integration."""

    def __init__(self):
        self.app = FastAPI(
            title="Swarms Agent API",
            description="Production-grade API for Swarms agent interaction",
            version="1.0.0",
            docs_url="/v1/docs",
            redoc_url="/v1/redoc",
        )
        self.store = AgentStore()
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "*"
            ],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._setup_routes()

    def _setup_routes(self):
        """Set up API routes."""

        @self.app.post("/v1/agent", response_model=Dict[str, UUID])
        async def create_agent(config: AgentConfig):
            """Create a new agent with the specified configuration."""
            agent_id = await self.store.create_agent(config)
            return {"agent_id": agent_id}

        @self.app.get("/v1/agents", response_model=List[AgentSummary])
        async def list_agents(
            tags: Optional[List[str]] = Query(None),
            status: Optional[AgentStatus] = None,
        ):
            """List all agents, optionally filtered by tags and status."""
            return await self.store.list_agents(tags, status)

        @self.app.patch(
            "/v1/agent/{agent_id}", response_model=Dict[str, str]
        )
        async def update_agent(agent_id: UUID, update: AgentUpdate):
            """Update an existing agent's configuration."""
            await self.store.update_agent(agent_id, update)
            return {"status": "updated"}

        @self.app.get(
            "/v1/agent/{agent_id}/metrics",
            response_model=AgentMetrics,
        )
        async def get_agent_metrics(agent_id: UUID):
            """Get performance metrics for a specific agent."""
            return await self.store.get_agent_metrics(agent_id)

        @self.app.post(
            "/v1/agent/{agent_id}/clone",
            response_model=Dict[str, UUID],
        )
        async def clone_agent(agent_id: UUID, new_name: str):
            """Clone an existing agent with a new name."""
            new_id = await self.store.clone_agent(agent_id, new_name)
            return {"agent_id": new_id}

        @self.app.delete("/v1/agent/{agent_id}")
        async def delete_agent(agent_id: UUID):
            """Delete an agent."""
            await self.store.delete_agent(agent_id)
            return {"status": "deleted"}

        @self.app.post(
            "/v1/agent/completions", response_model=CompletionResponse
        )
        async def create_completion(
            request: CompletionRequest,
            background_tasks: BackgroundTasks,
        ):
            """Process a completion request with the specified agent."""
            try:
                agent = await self.store.get_agent(request.agent_id)

                # Process completion
                response = await self.store.process_completion(
                    agent,
                    request.prompt,
                    request.agent_id,
                    request.max_tokens,
                    request.temperature_override,
                )

                # Schedule background cleanup
                background_tasks.add_task(
                    self._cleanup_old_metrics, request.agent_id
                )

                return response

            except Exception as e:
                logger.error(f"Error processing completion: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error processing completion: {str(e)}",
                )

        @self.app.get("/v1/agent/{agent_id}/status")
        async def get_agent_status(agent_id: UUID):
            """Get the current status of an agent."""
            metadata = self.store.agent_metadata.get(agent_id)
            if not metadata:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Agent {agent_id} not found",
                )
            return {
                "agent_id": agent_id,
                "status": metadata["status"],
                "last_used": metadata["last_used"],
                "total_completions": metadata["total_completions"],
                "error_count": metadata["error_count"],
            }

    async def _cleanup_old_metrics(self, agent_id: UUID):
        """Clean up old metrics data to prevent memory bloat."""
        metadata = self.store.agent_metadata.get(agent_id)
        if metadata:
            # Keep only last 24 hours of response times
            cutoff = datetime.utcnow() - timedelta(days=1)
            metadata["response_times"] = [
                t
                for t in metadata["response_times"]
                if isinstance(t, (int, float))
                and t > cutoff.timestamp()
            ]

            # Clean up old tokens per minute data
            if "tokens_per_minute" in metadata:
                metadata["tokens_per_minute"] = {
                    k: v
                    for k, v in metadata["tokens_per_minute"].items()
                    if k > cutoff
                }


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    api = SwarmsAPI()
    return api.app


if __name__ == "__main__":
    # Configure uvicorn logging
    logger.info("API Starting")
    uvicorn.run(
        "main:create_app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=4,
    )
