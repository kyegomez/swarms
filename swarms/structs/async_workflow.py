import asyncio
import json
import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from swarms.structs.agent import Agent
from swarms.structs.base_workflow import BaseWorkflow
from swarms.utils.loguru_logger import initialize_logger

# Base logger initialization
logger = initialize_logger("async_workflow")


# Pydantic models for structured data
class AgentOutput(BaseModel):
    agent_id: str
    agent_name: str
    task_id: str
    input: str
    output: Any
    start_time: datetime
    end_time: datetime
    status: str
    error: Optional[str] = None


class WorkflowOutput(BaseModel):
    workflow_id: str
    workflow_name: str
    start_time: datetime
    end_time: datetime
    total_agents: int
    successful_tasks: int
    failed_tasks: int
    agent_outputs: List[AgentOutput]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SpeakerRole(str, Enum):
    COORDINATOR = "coordinator"
    CRITIC = "critic"
    EXECUTOR = "executor"
    VALIDATOR = "validator"
    DEFAULT = "default"


class SpeakerMessage(BaseModel):
    role: SpeakerRole
    content: Any
    timestamp: datetime
    agent_name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GroupChatConfig(BaseModel):
    max_loops: int = 10
    timeout_per_turn: float = 30.0
    require_all_speakers: bool = False
    allow_concurrent: bool = True
    save_history: bool = True


@dataclass
class SharedMemoryItem:
    key: str
    value: Any
    timestamp: datetime
    author: str
    metadata: Dict[str, Any] = None


@dataclass
class SpeakerConfig:
    role: SpeakerRole
    agent: Any
    priority: int = 0
    concurrent: bool = True
    timeout: float = 30.0
    required: bool = False


class SharedMemory:
    """Thread-safe shared memory implementation with persistence"""

    def __init__(self, persistence_path: Optional[str] = None):
        self._memory = {}
        self._lock = threading.Lock()
        self._persistence_path = persistence_path
        self._load_from_disk()

    def set(
        self,
        key: str,
        value: Any,
        author: str,
        metadata: Dict[str, Any] = None,
    ) -> None:
        with self._lock:
            item = SharedMemoryItem(
                key=key,
                value=value,
                timestamp=datetime.utcnow(),
                author=author,
                metadata=metadata or {},
            )
            self._memory[key] = item
            self._persist_to_disk()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._memory.get(key)
            return item.value if item else None

    def get_with_metadata(
        self, key: str
    ) -> Optional[SharedMemoryItem]:
        with self._lock:
            return self._memory.get(key)

    def _persist_to_disk(self) -> None:
        if self._persistence_path:
            with open(self._persistence_path, "w") as f:
                json.dump(
                    {k: asdict(v) for k, v in self._memory.items()}, f
                )

    def _load_from_disk(self) -> None:
        if self._persistence_path and os.path.exists(
            self._persistence_path
        ):
            with open(self._persistence_path, "r") as f:
                data = json.load(f)
                self._memory = {
                    k: SharedMemoryItem(**v) for k, v in data.items()
                }


class SpeakerSystem:
    """Manages speaker interactions and group chat functionality"""

    def __init__(self, default_timeout: float = 30.0):
        self.speakers: Dict[SpeakerRole, SpeakerConfig] = {}
        self.message_history: List[SpeakerMessage] = []
        self.default_timeout = default_timeout
        self._lock = threading.Lock()

    def add_speaker(self, config: SpeakerConfig) -> None:
        with self._lock:
            self.speakers[config.role] = config

    def remove_speaker(self, role: SpeakerRole) -> None:
        with self._lock:
            self.speakers.pop(role, None)

    async def _execute_speaker(
        self,
        config: SpeakerConfig,
        input_data: Any,
        context: Dict[str, Any] = None,
    ) -> SpeakerMessage:
        try:
            result = await asyncio.wait_for(
                config.agent.arun(input_data), timeout=config.timeout
            )

            return SpeakerMessage(
                role=config.role,
                content=result,
                timestamp=datetime.utcnow(),
                agent_name=config.agent.agent_name,
                metadata={"context": context or {}},
            )
        except asyncio.TimeoutError:
            return SpeakerMessage(
                role=config.role,
                content=None,
                timestamp=datetime.utcnow(),
                agent_name=config.agent.agent_name,
                metadata={"error": "Timeout"},
            )
        except Exception as e:
            return SpeakerMessage(
                role=config.role,
                content=None,
                timestamp=datetime.utcnow(),
                agent_name=config.agent.agent_name,
                metadata={"error": str(e)},
            )


class AsyncWorkflow(BaseWorkflow):
    """Enhanced asynchronous workflow with advanced speaker system"""

    def __init__(
        self,
        name: str = "AsyncWorkflow",
        agents: List[Agent] = None,
        max_workers: int = 5,
        dashboard: bool = False,
        autosave: bool = False,
        verbose: bool = False,
        log_path: str = "workflow.log",
        shared_memory_path: Optional[str] = "shared_memory.json",
        enable_group_chat: bool = False,
        group_chat_config: Optional[GroupChatConfig] = None,
        **kwargs,
    ):
        super().__init__(agents=agents, **kwargs)
        self.workflow_id = str(uuid.uuid4())
        self.name = name
        self.agents = agents or []
        self.max_workers = max_workers
        self.dashboard = dashboard
        self.autosave = autosave
        self.verbose = verbose
        self.task_pool = []
        self.results = []
        self.shared_memory = SharedMemory(shared_memory_path)
        self.speaker_system = SpeakerSystem()
        self.enable_group_chat = enable_group_chat
        self.group_chat_config = (
            group_chat_config or GroupChatConfig()
        )
        self._setup_logging(log_path)
        self.metadata = {}

    def _setup_logging(self, log_path: str) -> None:
        """Configure rotating file logger"""
        self.logger = logging.getLogger(
            f"workflow_{self.workflow_id}"
        )
        self.logger.setLevel(
            logging.DEBUG if self.verbose else logging.INFO
        )

        handler = RotatingFileHandler(
            log_path, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def add_default_speakers(self) -> None:
        """Add all agents as default concurrent speakers"""
        for agent in self.agents:
            config = SpeakerConfig(
                role=SpeakerRole.DEFAULT,
                agent=agent,
                concurrent=True,
                timeout=30.0,
                required=False,
            )
            self.speaker_system.add_speaker(config)

    async def run_concurrent_speakers(
        self, task: str, context: Dict[str, Any] = None
    ) -> List[SpeakerMessage]:
        """Run all concurrent speakers in parallel"""
        concurrent_tasks = [
            self.speaker_system._execute_speaker(
                config, task, context
            )
            for config in self.speaker_system.speakers.values()
            if config.concurrent
        ]

        results = await asyncio.gather(
            *concurrent_tasks, return_exceptions=True
        )
        return [r for r in results if isinstance(r, SpeakerMessage)]

    async def run_sequential_speakers(
        self, task: str, context: Dict[str, Any] = None
    ) -> List[SpeakerMessage]:
        """Run non-concurrent speakers in sequence"""
        results = []
        for config in sorted(
            self.speaker_system.speakers.values(),
            key=lambda x: x.priority,
        ):
            if not config.concurrent:
                result = await self.speaker_system._execute_speaker(
                    config, task, context
                )
                results.append(result)
        return results

    async def run_group_chat(
        self, initial_message: str, context: Dict[str, Any] = None
    ) -> List[SpeakerMessage]:
        """Run a group chat discussion among speakers"""
        if not self.enable_group_chat:
            raise ValueError(
                "Group chat is not enabled for this workflow"
            )

        messages: List[SpeakerMessage] = []
        current_turn = 0

        while current_turn < self.group_chat_config.max_loops:
            turn_context = {
                "turn": current_turn,
                "history": messages,
                **(context or {}),
            }

            if self.group_chat_config.allow_concurrent:
                turn_messages = await self.run_concurrent_speakers(
                    (
                        initial_message
                        if current_turn == 0
                        else messages[-1].content
                    ),
                    turn_context,
                )
            else:
                turn_messages = await self.run_sequential_speakers(
                    (
                        initial_message
                        if current_turn == 0
                        else messages[-1].content
                    ),
                    turn_context,
                )

            messages.extend(turn_messages)

            # Check if we should continue the conversation
            if self._should_end_group_chat(messages):
                break

            current_turn += 1

        if self.group_chat_config.save_history:
            self.speaker_system.message_history.extend(messages)

        return messages

    def _should_end_group_chat(
        self, messages: List[SpeakerMessage]
    ) -> bool:
        """Determine if group chat should end based on messages"""
        if not messages:
            return True

        # Check if all required speakers have participated
        if self.group_chat_config.require_all_speakers:
            participating_roles = {msg.role for msg in messages}
            required_roles = {
                role
                for role, config in self.speaker_system.speakers.items()
                if config.required
            }
            if not required_roles.issubset(participating_roles):
                return False

        return False

    @asynccontextmanager
    async def task_context(self):
        """Context manager for task execution with proper cleanup"""
        start_time = datetime.utcnow()
        try:
            yield
        finally:
            end_time = datetime.utcnow()
            if self.autosave:
                await self._save_results(start_time, end_time)

    async def _execute_agent_task(
        self, agent: Agent, task: str
    ) -> AgentOutput:
        """Execute a single agent task with enhanced error handling and monitoring"""
        start_time = datetime.utcnow()
        task_id = str(uuid.uuid4())

        try:
            self.logger.info(
                f"Agent {agent.agent_name} starting task {task_id}: {task}"
            )

            result = await agent.arun(task)

            end_time = datetime.utcnow()
            self.logger.info(
                f"Agent {agent.agent_name} completed task {task_id}"
            )

            return AgentOutput(
                agent_id=str(id(agent)),
                agent_name=agent.agent_name,
                task_id=task_id,
                input=task,
                output=result,
                start_time=start_time,
                end_time=end_time,
                status="success",
            )

        except Exception as e:
            end_time = datetime.utcnow()
            self.logger.error(
                f"Error in agent {agent.agent_name} task {task_id}: {str(e)}",
                exc_info=True,
            )

            return AgentOutput(
                agent_id=str(id(agent)),
                agent_name=agent.agent_name,
                task_id=task_id,
                input=task,
                output=None,
                start_time=start_time,
                end_time=end_time,
                status="error",
                error=str(e),
            )

    async def run(self, task: str) -> WorkflowOutput:
        """Enhanced workflow execution with speaker system integration"""
        if not self.agents:
            raise ValueError("No agents provided to the workflow")

        async with self.task_context():
            start_time = datetime.utcnow()

            try:
                # Run speakers first if enabled
                speaker_outputs = []
                if self.enable_group_chat:
                    speaker_outputs = await self.run_group_chat(task)
                else:
                    concurrent_outputs = (
                        await self.run_concurrent_speakers(task)
                    )
                    sequential_outputs = (
                        await self.run_sequential_speakers(task)
                    )
                    speaker_outputs = (
                        concurrent_outputs + sequential_outputs
                    )

                # Store speaker outputs in shared memory
                self.shared_memory.set(
                    "speaker_outputs",
                    [msg.dict() for msg in speaker_outputs],
                    "workflow",
                )

                # Create tasks for all agents
                tasks = [
                    self._execute_agent_task(agent, task)
                    for agent in self.agents
                ]

                # Execute all tasks concurrently
                agent_outputs = await asyncio.gather(
                    *tasks, return_exceptions=True
                )

                end_time = datetime.utcnow()

                # Calculate success/failure counts
                successful_tasks = sum(
                    1
                    for output in agent_outputs
                    if isinstance(output, AgentOutput)
                    and output.status == "success"
                )
                failed_tasks = len(agent_outputs) - successful_tasks

                return WorkflowOutput(
                    workflow_id=self.workflow_id,
                    workflow_name=self.name,
                    start_time=start_time,
                    end_time=end_time,
                    total_agents=len(self.agents),
                    successful_tasks=successful_tasks,
                    failed_tasks=failed_tasks,
                    agent_outputs=[
                        output
                        for output in agent_outputs
                        if isinstance(output, AgentOutput)
                    ],
                    metadata={
                        "max_workers": self.max_workers,
                        "shared_memory_keys": list(
                            self.shared_memory._memory.keys()
                        ),
                        "group_chat_enabled": self.enable_group_chat,
                        "total_speaker_messages": len(
                            speaker_outputs
                        ),
                        "speaker_outputs": [
                            msg.dict() for msg in speaker_outputs
                        ],
                    },
                )

            except Exception as e:
                self.logger.error(
                    f"Critical workflow error: {str(e)}",
                    exc_info=True,
                )
                raise

    async def _save_results(
        self, start_time: datetime, end_time: datetime
    ) -> None:
        """Save workflow results to disk"""
        if not self.autosave:
            return

        output_dir = "workflow_outputs"
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{output_dir}/workflow_{self.workflow_id}_{end_time.strftime('%Y%m%d_%H%M%S')}.json"

        try:
            with open(filename, "w") as f:
                json.dump(
                    {
                        "workflow_id": self.workflow_id,
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "results": [
                            (
                                asdict(result)
                                if hasattr(result, "__dict__")
                                else (
                                    result.dict()
                                    if hasattr(result, "dict")
                                    else str(result)
                                )
                            )
                            for result in self.results
                        ],
                        "speaker_history": [
                            msg.dict()
                            for msg in self.speaker_system.message_history
                        ],
                        "metadata": self.metadata,
                    },
                    f,
                    default=str,
                    indent=2,
                )

            self.logger.info(f"Workflow results saved to {filename}")
        except Exception as e:
            self.logger.error(
                f"Error saving workflow results: {str(e)}"
            )

    def _validate_config(self) -> None:
        """Validate workflow configuration"""
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        if (
            self.enable_group_chat
            and not self.speaker_system.speakers
        ):
            raise ValueError(
                "Group chat enabled but no speakers configured"
            )

        for config in self.speaker_system.speakers.values():
            if config.timeout <= 0:
                raise ValueError(
                    f"Invalid timeout for speaker {config.role}"
                )

    async def cleanup(self) -> None:
        """Cleanup workflow resources"""
        try:
            # Close any open file handlers
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)

            # Persist final state
            if self.autosave:
                end_time = datetime.utcnow()
                await self._save_results(
                    (
                        self.results[0].start_time
                        if self.results
                        else end_time
                    ),
                    end_time,
                )

            # Clear shared memory if configured
            self.shared_memory._memory.clear()

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise


# Utility functions for the workflow
def create_default_workflow(
    agents: List[Agent],
    name: str = "DefaultWorkflow",
    enable_group_chat: bool = False,
) -> AsyncWorkflow:
    """Create a workflow with default configuration"""
    workflow = AsyncWorkflow(
        name=name,
        agents=agents,
        max_workers=len(agents),
        dashboard=True,
        autosave=True,
        verbose=True,
        enable_group_chat=enable_group_chat,
        group_chat_config=GroupChatConfig(
            max_loops=5,
            allow_concurrent=True,
            require_all_speakers=False,
        ),
    )

    workflow.add_default_speakers()
    return workflow


async def run_workflow_with_retry(
    workflow: AsyncWorkflow,
    task: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> WorkflowOutput:
    """Run workflow with retry logic"""
    for attempt in range(max_retries):
        try:
            return await workflow.run(task)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            workflow.logger.warning(
                f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds: {str(e)}"
            )
            await asyncio.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff


# async def create_specialized_agents() -> List[Agent]:
#     """Create a set of specialized agents for financial analysis"""

#     # Base model configuration
#     model = OpenAIChat(model_name="gpt-4o")

#     # Financial Analysis Agent
#     financial_agent = Agent(
#         agent_name="Financial-Analysis-Agent",
#         agent_description="Personal finance advisor agent",
#         system_prompt=FINANCIAL_AGENT_SYS_PROMPT +
#             "Output the <DONE> token when you're done creating a portfolio of etfs, index, funds, and more for AI",
#         max_loops=1,
#         llm=model,
#         dynamic_temperature_enabled=True,
#         user_name="Kye",
#         retry_attempts=3,
#         context_length=8192,
#         return_step_meta=False,
#         output_type="str",
#         auto_generate_prompt=False,
#         max_tokens=4000,
#         stopping_token="<DONE>",
#         saved_state_path="financial_agent.json",
#         interactive=False,
#     )

#     # Risk Assessment Agent
#     risk_agent = Agent(
#         agent_name="Risk-Assessment-Agent",
#         agent_description="Investment risk analysis specialist",
#         system_prompt="Analyze investment risks and provide risk scores. Output <DONE> when analysis is complete.",
#         max_loops=1,
#         llm=model,
#         dynamic_temperature_enabled=True,
#         user_name="Kye",
#         retry_attempts=3,
#         context_length=8192,
#         output_type="str",
#         max_tokens=4000,
#         stopping_token="<DONE>",
#         saved_state_path="risk_agent.json",
#         interactive=False,
#     )

#     # Market Research Agent
#     research_agent = Agent(
#         agent_name="Market-Research-Agent",
#         agent_description="AI and tech market research specialist",
#         system_prompt="Research AI market trends and growth opportunities. Output <DONE> when research is complete.",
#         max_loops=1,
#         llm=model,
#         dynamic_temperature_enabled=True,
#         user_name="Kye",
#         retry_attempts=3,
#         context_length=8192,
#         output_type="str",
#         max_tokens=4000,
#         stopping_token="<DONE>",
#         saved_state_path="research_agent.json",
#         interactive=False,
#     )

#     return [financial_agent, risk_agent, research_agent]

# async def main():
#     # Create specialized agents
#     agents = await create_specialized_agents()

#     # Create workflow with group chat enabled
#     workflow = create_default_workflow(
#         agents=agents,
#         name="AI-Investment-Analysis-Workflow",
#         enable_group_chat=True
#     )

#     # Configure speaker roles
#     workflow.speaker_system.add_speaker(
#         SpeakerConfig(
#             role=SpeakerRole.COORDINATOR,
#             agent=agents[0],  # Financial agent as coordinator
#             priority=1,
#             concurrent=False,
#             required=True
#         )
#     )

#     workflow.speaker_system.add_speaker(
#         SpeakerConfig(
#             role=SpeakerRole.CRITIC,
#             agent=agents[1],  # Risk agent as critic
#             priority=2,
#             concurrent=True
#         )
#     )

#     workflow.speaker_system.add_speaker(
#         SpeakerConfig(
#             role=SpeakerRole.EXECUTOR,
#             agent=agents[2],  # Research agent as executor
#             priority=2,
#             concurrent=True
#         )
#     )

#     # Investment analysis task
#     investment_task = """
#     Create a comprehensive investment analysis for a $40k portfolio focused on AI growth opportunities:
#     1. Identify high-growth AI ETFs and index funds
#     2. Analyze risks and potential returns
#     3. Create a diversified portfolio allocation
#     4. Provide market trend analysis
#     Present the results in a structured markdown format.
#     """

#     try:
#         # Run workflow with retry
#         result = await run_workflow_with_retry(
#             workflow=workflow,
#             task=investment_task,
#             max_retries=3
#         )

#         print("\nWorkflow Results:")
#         print("================")

#         # Process and display agent outputs
#         for output in result.agent_outputs:
#             print(f"\nAgent: {output.agent_name}")
#             print("-" * (len(output.agent_name) + 8))
#             print(output.output)

#         # Display group chat history if enabled
#         if workflow.enable_group_chat:
#             print("\nGroup Chat Discussion:")
#             print("=====================")
#             for msg in workflow.speaker_system.message_history:
#                 print(f"\n{msg.role} ({msg.agent_name}):")
#                 print(msg.content)

#         # Save detailed results
#         if result.metadata.get("shared_memory_keys"):
#             print("\nShared Insights:")
#             print("===============")
#             for key in result.metadata["shared_memory_keys"]:
#                 value = workflow.shared_memory.get(key)
#                 if value:
#                     print(f"\n{key}:")
#                     print(value)

#     except Exception as e:
#         print(f"Workflow failed: {str(e)}")

#     finally:
#         await workflow.cleanup()

# if __name__ == "__main__":
#     # Run the example
#     asyncio.run(main())
