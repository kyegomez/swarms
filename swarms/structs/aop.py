import asyncio
import socket
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from loguru import logger
from mcp.server.fastmcp import FastMCP

from swarms.structs.agent import Agent
from swarms.structs.omni_agent_types import AgentType
from swarms.tools.mcp_client_tools import (
    get_tools_for_multiple_mcp_servers,
)


class TaskStatus(Enum):
    """Status of a task in the queue."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QueueStatus(Enum):
    """Status of a task queue."""

    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class Task:
    """
    Represents a task to be executed by an agent.

    Attributes:
        task_id: Unique identifier for the task
        task: The task or prompt to execute
        img: Optional image to be processed
        imgs: Optional list of images to be processed
        correct_answer: Optional correct answer for validation
        priority: Task priority (higher number = higher priority)
        created_at: Timestamp when task was created
        status: Current status of the task
        result: Result of task execution
        error: Error message if task failed
        retry_count: Number of times task has been retried
        max_retries: Maximum number of retries allowed
    """

    task_id: str = field(default_factory=lambda: str(uuid4()))
    task: str = ""
    img: Optional[str] = None
    imgs: Optional[List[str]] = None
    correct_answer: Optional[str] = None
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class QueueStats:
    """
    Statistics for a task queue.

    Attributes:
        total_tasks: Total number of tasks processed
        completed_tasks: Number of successfully completed tasks
        failed_tasks: Number of failed tasks
        pending_tasks: Number of tasks currently pending
        processing_tasks: Number of tasks currently being processed
        average_processing_time: Average time to process a task
        queue_size: Current size of the queue
    """

    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    pending_tasks: int = 0
    processing_tasks: int = 0
    average_processing_time: float = 0.0
    queue_size: int = 0


class TaskQueue:
    """
    A thread-safe task queue for managing agent tasks.

    This class provides functionality to:
    1. Add tasks to the queue with priority support
    2. Process tasks in background workers
    3. Handle task retries and error management
    4. Provide queue statistics and monitoring
    """

    def __init__(
        self,
        agent_name: str,
        agent: AgentType,
        max_workers: int = 1,
        max_queue_size: int = 1000,
        processing_timeout: int = 30,
        retry_delay: float = 1.0,
        verbose: bool = False,
    ):
        """
        Initialize the task queue.

        Args:
            agent_name: Name of the agent this queue belongs to
            agent: The agent instance to execute tasks
            max_workers: Maximum number of worker threads
            max_queue_size: Maximum number of tasks in queue
            processing_timeout: Timeout for task processing in seconds
            retry_delay: Delay between retries in seconds
            verbose: Enable verbose logging
        """
        self.agent_name = agent_name
        self.agent = agent
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.processing_timeout = processing_timeout
        self.retry_delay = retry_delay
        self.verbose = verbose

        # Queue management
        self._queue = deque()
        self._lock = threading.RLock()
        self._status = QueueStatus.STOPPED
        self._workers = []
        self._stop_event = threading.Event()

        # Statistics
        self._stats = QueueStats()
        self._processing_times = deque(
            maxlen=100
        )  # Keep last 100 processing times

        # Task tracking
        self._tasks = {}  # task_id -> Task
        self._processing_tasks = (
            set()
        )  # Currently processing task IDs

        logger.info(
            f"Initialized TaskQueue for agent '{agent_name}' with {max_workers} workers"
        )

    def add_task(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        correct_answer: Optional[str] = None,
        priority: int = 0,
        max_retries: int = 3,
    ) -> str:
        """
        Add a task to the queue.

        Args:
            task: The task or prompt to execute
            img: Optional image to be processed
            imgs: Optional list of images to be processed
            correct_answer: Optional correct answer for validation
            priority: Task priority (higher number = higher priority)
            max_retries: Maximum number of retries allowed

        Returns:
            str: Task ID

        Raises:
            ValueError: If queue is full or task is invalid
        """
        if not task:
            raise ValueError("Task cannot be empty")

        with self._lock:
            if len(self._queue) >= self.max_queue_size:
                raise ValueError(
                    f"Queue is full (max size: {self.max_queue_size})"
                )

            task_obj = Task(
                task=task,
                img=img,
                imgs=imgs,
                correct_answer=correct_answer,
                priority=priority,
                max_retries=max_retries,
            )

            # Insert task based on priority (higher priority first)
            inserted = False
            for i, existing_task in enumerate(self._queue):
                if task_obj.priority > existing_task.priority:
                    self._queue.insert(i, task_obj)
                    inserted = True
                    break

            if not inserted:
                self._queue.append(task_obj)

            self._tasks[task_obj.task_id] = task_obj
            self._stats.total_tasks += 1
            self._stats.pending_tasks += 1
            self._stats.queue_size = len(self._queue)

            if self.verbose:
                logger.debug(
                    f"Added task '{task_obj.task_id}' to queue for agent '{self.agent_name}'"
                )

            return task_obj.task_id

    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task by ID.

        Args:
            task_id: The task ID

        Returns:
            Task object or None if not found
        """
        with self._lock:
            return self._tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.

        Args:
            task_id: The task ID to cancel

        Returns:
            bool: True if task was cancelled, False if not found or already processed
        """
        with self._lock:
            if task_id not in self._tasks:
                return False

            task = self._tasks[task_id]
            if task.status in [
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            ]:
                return False

            # Remove from queue if still pending
            if task.status == TaskStatus.PENDING:
                try:
                    self._queue.remove(task)
                    self._stats.pending_tasks -= 1
                    self._stats.queue_size = len(self._queue)
                except ValueError:
                    pass  # Task not in queue

            # Mark as cancelled
            task.status = TaskStatus.CANCELLED
            self._processing_tasks.discard(task_id)

            if self.verbose:
                logger.debug(
                    f"Cancelled task '{task_id}' for agent '{self.agent_name}'"
                )

            return True

    def start_workers(self) -> None:
        """Start the background worker threads."""
        with self._lock:
            if self._status != QueueStatus.STOPPED:
                logger.warning(
                    f"Workers for agent '{self.agent_name}' are already running"
                )
                return

            self._status = QueueStatus.RUNNING
            self._stop_event.clear()

            for i in range(self.max_workers):
                worker = threading.Thread(
                    target=self._worker_loop,
                    name=f"Worker-{self.agent_name}-{i}",
                    daemon=True,
                )
                worker.start()
                self._workers.append(worker)

            logger.info(
                f"Started {self.max_workers} workers for agent '{self.agent_name}'"
            )

    def stop_workers(self) -> None:
        """Stop the background worker threads."""
        with self._lock:
            if self._status == QueueStatus.STOPPED:
                return

            self._status = QueueStatus.STOPPED
            self._stop_event.set()

            # Wait for workers to finish
            for worker in self._workers:
                worker.join(timeout=5.0)

            self._workers.clear()
            logger.info(
                f"Stopped workers for agent '{self.agent_name}'"
            )

    def pause_workers(self) -> None:
        """Pause the workers (they will finish current tasks but not start new ones)."""
        with self._lock:
            if self._status == QueueStatus.RUNNING:
                self._status = QueueStatus.PAUSED
                logger.info(
                    f"Paused workers for agent '{self.agent_name}'"
                )

    def resume_workers(self) -> None:
        """Resume the workers."""
        with self._lock:
            if self._status == QueueStatus.PAUSED:
                self._status = QueueStatus.RUNNING
                logger.info(
                    f"Resumed workers for agent '{self.agent_name}'"
                )

    def clear_queue(self) -> int:
        """
        Clear all pending tasks from the queue.

        Returns:
            int: Number of tasks cleared
        """
        with self._lock:
            cleared_count = len(self._queue)
            self._queue.clear()
            self._stats.pending_tasks = 0
            self._stats.queue_size = 0

            # Mark all pending tasks as cancelled
            for task in self._tasks.values():
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.CANCELLED

            if self.verbose:
                logger.debug(
                    f"Cleared {cleared_count} tasks from queue for agent '{self.agent_name}'"
                )

            return cleared_count

    def get_stats(self) -> QueueStats:
        """Get current queue statistics."""
        with self._lock:
            # Update current stats
            self._stats.pending_tasks = len(
                [
                    t
                    for t in self._tasks.values()
                    if t.status == TaskStatus.PENDING
                ]
            )
            self._stats.processing_tasks = len(self._processing_tasks)
            self._stats.queue_size = len(self._queue)

            # Calculate average processing time
            if self._processing_times:
                self._stats.average_processing_time = sum(
                    self._processing_times
                ) / len(self._processing_times)

            return QueueStats(
                total_tasks=self._stats.total_tasks,
                completed_tasks=self._stats.completed_tasks,
                failed_tasks=self._stats.failed_tasks,
                pending_tasks=self._stats.pending_tasks,
                processing_tasks=self._stats.processing_tasks,
                average_processing_time=self._stats.average_processing_time,
                queue_size=self._stats.queue_size,
            )

    def get_status(self) -> QueueStatus:
        """Get current queue status."""
        return self._status

    def _worker_loop(self) -> None:
        """Main worker loop for processing tasks."""
        while not self._stop_event.is_set():
            try:
                # Check if we should process tasks
                with self._lock:
                    if (
                        self._status != QueueStatus.RUNNING
                        or not self._queue
                    ):
                        self._stop_event.wait(0.1)
                        continue

                    # Get next task
                    task = self._queue.popleft()
                    self._processing_tasks.add(task.task_id)
                    task.status = TaskStatus.PROCESSING
                    self._stats.pending_tasks -= 1
                    self._stats.processing_tasks += 1

                # Process the task
                self._process_task(task)

            except Exception as e:
                logger.error(
                    f"Error in worker loop for agent '{self.agent_name}': {e}"
                )
                if self.verbose:
                    logger.error(traceback.format_exc())
                time.sleep(0.1)

    def _process_task(self, task: Task) -> None:
        """
        Process a single task.

        Args:
            task: The task to process
        """
        start_time = time.time()

        try:
            if self.verbose:
                logger.debug(
                    f"Processing task '{task.task_id}' for agent '{self.agent_name}'"
                )

            # Execute the agent
            result = self.agent.run(
                task=task.task,
                img=task.img,
                imgs=task.imgs,
                correct_answer=task.correct_answer,
            )

            # Update task with result
            task.result = result
            task.status = TaskStatus.COMPLETED

            # Update statistics
            processing_time = time.time() - start_time
            self._processing_times.append(processing_time)

            with self._lock:
                self._stats.completed_tasks += 1
                self._stats.processing_tasks -= 1
                self._processing_tasks.discard(task.task_id)

            if self.verbose:
                logger.debug(
                    f"Completed task '{task.task_id}' in {processing_time:.2f}s"
                )

        except Exception as e:
            error_msg = str(e)
            task.error = error_msg
            task.retry_count += 1

            if self.verbose:
                logger.error(
                    f"Error processing task '{task.task_id}': {error_msg}"
                )
                logger.error(traceback.format_exc())

            # Handle retries
            if task.retry_count <= task.max_retries:
                if self.verbose:
                    logger.debug(
                        f"Retrying task '{task.task_id}' (attempt {task.retry_count + 1})"
                    )

                # Re-queue the task with a delay
                time.sleep(self.retry_delay)

                with self._lock:
                    if self._status == QueueStatus.RUNNING:
                        task.status = TaskStatus.PENDING
                        self._queue.append(
                            task
                        )  # Add to end of queue
                        self._stats.pending_tasks += 1
                        self._stats.queue_size = len(self._queue)
                    else:
                        task.status = TaskStatus.FAILED
                        self._stats.failed_tasks += 1
            else:
                # Max retries exceeded
                task.status = TaskStatus.FAILED

                with self._lock:
                    self._stats.failed_tasks += 1
                    self._stats.processing_tasks -= 1
                    self._processing_tasks.discard(task.task_id)

                if self.verbose:
                    logger.error(
                        f"Task '{task.task_id}' failed after {task.max_retries} retries"
                    )


@dataclass
class AgentToolConfig:
    """
    Configuration for converting an agent to an MCP tool.

    Attributes:
        tool_name: The name of the tool in the MCP server
        tool_description: Description of what the tool does
        input_schema: JSON schema for the tool's input parameters
        output_schema: JSON schema for the tool's output
        timeout: Maximum time to wait for agent execution (seconds)
        max_retries: Number of retries if agent execution fails
        verbose: Enable verbose logging for this tool
        traceback_enabled: Enable traceback logging for errors
    """

    tool_name: str
    tool_description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    timeout: int = 30
    max_retries: int = 3
    verbose: bool = False
    traceback_enabled: bool = True


class AOP:
    """
    A class that takes a list of agents and deploys them as unique tools in an MCP server.

    This class provides functionality to:
    1. Convert swarms agents into MCP tools
    2. Deploy multiple agents as individual tools
    3. Handle tool execution with proper error handling
    4. Manage the MCP server lifecycle
    5. Queue-based task execution for improved performance and reliability
    6. Persistence mode with automatic restart and failsafe protection

    Attributes:
        mcp_server: The FastMCP server instance
        agents: Dictionary mapping tool names to agent instances
        tool_configs: Dictionary mapping tool names to their configurations
        task_queues: Dictionary mapping tool names to their task queues
        server_name: Name of the MCP server
        queue_enabled: Whether queue-based execution is enabled
        persistence: Whether persistence mode is enabled
        max_restart_attempts: Maximum number of restart attempts before giving up
        restart_delay: Delay between restart attempts in seconds
        network_monitoring: Whether network connection monitoring is enabled
        max_network_retries: Maximum number of network reconnection attempts
        network_retry_delay: Delay between network retry attempts in seconds
        network_timeout: Network connection timeout in seconds
    """

    def __init__(
        self,
        server_name: str = "AOP Cluster",
        description: str = "A cluster that enables you to deploy multiple agents as tools in an MCP server.",
        agents: any = None,
        port: int = 8000,
        transport: str = "streamable-http",
        verbose: bool = False,
        traceback_enabled: bool = True,
        host: str = "localhost",
        queue_enabled: bool = True,
        max_workers_per_agent: int = 1,
        max_queue_size_per_agent: int = 1000,
        processing_timeout: int = 30,
        retry_delay: float = 1.0,
        persistence: bool = False,
        max_restart_attempts: int = 10,
        restart_delay: float = 5.0,
        network_monitoring: bool = True,
        max_network_retries: int = 5,
        network_retry_delay: float = 10.0,
        network_timeout: float = 30.0,
        log_level: Literal[
            "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        ] = "INFO",
        *args,
        **kwargs,
    ):
        """
        Initialize the AOP.

        Args:
            server_name: Name for the MCP server
            description: Description of the AOP cluster
            agents: Optional list of agents to add initially
            port: Port for the MCP server
            transport: Transport type for the MCP server
            verbose: Enable verbose logging
            traceback_enabled: Enable traceback logging for errors
            host: Host to bind the server to
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            queue_enabled: Enable queue-based task execution
            max_workers_per_agent: Maximum number of workers per agent
            max_queue_size_per_agent: Maximum queue size per agent
            processing_timeout: Timeout for task processing in seconds
            retry_delay: Delay between retries in seconds
            persistence: Enable automatic restart on shutdown (with failsafe)
            max_restart_attempts: Maximum number of restart attempts before giving up
            restart_delay: Delay between restart attempts in seconds
            network_monitoring: Enable network connection monitoring and retry
            max_network_retries: Maximum number of network reconnection attempts
            network_retry_delay: Delay between network retry attempts in seconds
            network_timeout: Network connection timeout in seconds
        """
        self.server_name = server_name
        self.description = description
        self.verbose = verbose
        self.traceback_enabled = traceback_enabled
        self.log_level = log_level
        self.host = host
        self.port = port
        self.queue_enabled = queue_enabled
        self.max_workers_per_agent = max_workers_per_agent
        self.max_queue_size_per_agent = max_queue_size_per_agent
        self.processing_timeout = processing_timeout
        self.retry_delay = retry_delay
        self.persistence = persistence
        self.max_restart_attempts = max_restart_attempts
        self.restart_delay = restart_delay
        self.network_monitoring = network_monitoring
        self.max_network_retries = max_network_retries
        self.network_retry_delay = network_retry_delay
        self.network_timeout = network_timeout

        # Persistence state tracking
        self._restart_count = 0
        self._persistence_enabled = persistence
        self._shutdown_requested = False

        # Network state tracking
        self._network_retry_count = 0
        self._last_network_error = None
        self._network_connected = True

        # Server creation timestamp
        self._created_at = time.time()

        self.agents: Dict[str, Agent] = {}
        self.tool_configs: Dict[str, AgentToolConfig] = {}
        self.task_queues: Dict[str, TaskQueue] = {}
        self.transport = transport
        self.mcp_server = FastMCP(
            name=server_name,
            port=port,
            log_level=log_level,
            *args,
            **kwargs,
        )

        # Configure logger
        logger.remove()  # Remove default handler
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True,
        )

        logger.info(
            f"Initialized AOP with server name: {server_name}, verbose: {verbose}, traceback: {traceback_enabled}, persistence: {persistence}, network_monitoring: {network_monitoring}"
        )

        # Add initial agents if provided
        if agents:
            logger.info(f"Adding {len(agents)} initial agents")
            self.add_agents_batch(agents)

        # Register the agent discovery tool
        self._register_agent_discovery_tool()

        # Register queue management tools if queue is enabled
        if self.queue_enabled:
            self._register_queue_management_tools()

    def add_agent(
        self,
        agent: AgentType,
        tool_name: Optional[str] = None,
        tool_description: Optional[str] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
        max_retries: int = 3,
        verbose: Optional[bool] = None,
        traceback_enabled: Optional[bool] = None,
    ) -> str:
        """
        Add an agent to the MCP server as a tool.

        Args:
            agent: The swarms Agent instance to deploy
            tool_name: Name for the tool (defaults to agent.agent_name)
            tool_description: Description of the tool (defaults to agent.agent_description)
            input_schema: JSON schema for input parameters
            output_schema: JSON schema for output
            timeout: Maximum execution time in seconds
            max_retries: Number of retries on failure
            verbose: Enable verbose logging for this tool (defaults to deployer's verbose setting)
            traceback_enabled: Enable traceback logging for this tool (defaults to deployer's setting)

        Returns:
            str: The tool name that was registered

        Raises:
            ValueError: If agent is None or tool_name already exists
        """
        if agent is None:
            logger.error("Cannot add None agent")
            raise ValueError("Agent cannot be None")

        # Use agent name as tool name if not provided
        if tool_name is None:
            tool_name = (
                agent.agent_name or f"agent_{len(self.agents)}"
            )

        if tool_name in self.agents:
            logger.error(f"Tool name '{tool_name}' already exists")
            raise ValueError(
                f"Tool name '{tool_name}' already exists"
            )

        # Use deployer defaults if not specified
        if verbose is None:
            verbose = self.verbose
        if traceback_enabled is None:
            traceback_enabled = self.traceback_enabled

        logger.debug(
            f"Adding agent '{agent.agent_name}' as tool '{tool_name}' with verbose={verbose}, traceback={traceback_enabled}"
        )

        # Use agent description if not provided
        if tool_description is None:
            tool_description = (
                agent.agent_description
                or f"Agent tool: {agent.agent_name}"
            )

        # Default input schema for task-based agents
        if input_schema is None:
            input_schema = {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task or prompt to execute with this agent",
                    },
                    "img": {
                        "type": "string",
                        "description": "Optional image to be processed by the agent",
                    },
                    "imgs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of images to be processed by the agent",
                    },
                    "correct_answer": {
                        "type": "string",
                        "description": "Optional correct answer for validation or comparison",
                    },
                },
                "required": ["task"],
            }

        # Default output schema
        if output_schema is None:
            output_schema = {
                "type": "object",
                "properties": {
                    "result": {
                        "type": "string",
                        "description": "The agent's response to the task",
                    },
                    "success": {
                        "type": "boolean",
                        "description": "Whether the task was executed successfully",
                    },
                    "error": {
                        "type": "string",
                        "description": "Error message if execution failed",
                    },
                },
                "required": ["result", "success"],
            }

        # Store agent and configuration
        self.agents[tool_name] = agent
        self.tool_configs[tool_name] = AgentToolConfig(
            tool_name=tool_name,
            tool_description=tool_description,
            input_schema=input_schema,
            output_schema=output_schema,
            timeout=timeout,
            max_retries=max_retries,
            verbose=verbose,
            traceback_enabled=traceback_enabled,
        )

        # Create task queue if queue is enabled
        if self.queue_enabled:
            self.task_queues[tool_name] = TaskQueue(
                agent_name=tool_name,
                agent=agent,
                max_workers=self.max_workers_per_agent,
                max_queue_size=self.max_queue_size_per_agent,
                processing_timeout=self.processing_timeout,
                retry_delay=self.retry_delay,
                verbose=verbose,
            )
            # Start the queue workers
            self.task_queues[tool_name].start_workers()

        # Register the tool with the MCP server
        self._register_tool(tool_name, agent)

        # Re-register the discovery tool to include the new agent
        self._register_agent_discovery_tool()

        logger.info(
            f"Added agent '{agent.agent_name}' as tool '{tool_name}' (verbose={verbose}, traceback={traceback_enabled}, queue_enabled={self.queue_enabled})"
        )
        return tool_name

    def add_agents_batch(
        self,
        agents: List[Agent],
        tool_names: Optional[List[str]] = None,
        tool_descriptions: Optional[List[str]] = None,
        input_schemas: Optional[List[Dict[str, Any]]] = None,
        output_schemas: Optional[List[Dict[str, Any]]] = None,
        timeouts: Optional[List[int]] = None,
        max_retries_list: Optional[List[int]] = None,
        verbose_list: Optional[List[bool]] = None,
        traceback_enabled_list: Optional[List[bool]] = None,
    ) -> List[str]:
        """
        Add multiple agents to the MCP server as tools in batch.

        Args:
            agents: List of swarms Agent instances
            tool_names: Optional list of tool names (defaults to agent names)
            tool_descriptions: Optional list of tool descriptions
            input_schemas: Optional list of input schemas
            output_schemas: Optional list of output schemas
            timeouts: Optional list of timeout values
            max_retries_list: Optional list of max retry values
            verbose_list: Optional list of verbose settings for each agent
            traceback_enabled_list: Optional list of traceback settings for each agent

        Returns:
            List[str]: List of tool names that were registered

        Raises:
            ValueError: If agents list is empty or contains None values
        """
        if not agents:
            logger.error("Cannot add empty agents list")
            raise ValueError("Agents list cannot be empty")

        if None in agents:
            logger.error("Agents list contains None values")
            raise ValueError("Agents list cannot contain None values")

        logger.info(f"Adding {len(agents)} agents in batch")
        registered_tools = []

        for i, agent in enumerate(agents):
            tool_name = (
                tool_names[i]
                if tool_names and i < len(tool_names)
                else None
            )
            tool_description = (
                tool_descriptions[i]
                if tool_descriptions and i < len(tool_descriptions)
                else None
            )
            input_schema = (
                input_schemas[i]
                if input_schemas and i < len(input_schemas)
                else None
            )
            output_schema = (
                output_schemas[i]
                if output_schemas and i < len(output_schemas)
                else None
            )
            timeout = (
                timeouts[i] if timeouts and i < len(timeouts) else 30
            )
            max_retries = (
                max_retries_list[i]
                if max_retries_list and i < len(max_retries_list)
                else 3
            )
            verbose = (
                verbose_list[i]
                if verbose_list and i < len(verbose_list)
                else None
            )
            traceback_enabled = (
                traceback_enabled_list[i]
                if traceback_enabled_list
                and i < len(traceback_enabled_list)
                else None
            )

            tool_name = self.add_agent(
                agent=agent,
                tool_name=tool_name,
                tool_description=tool_description,
                input_schema=input_schema,
                output_schema=output_schema,
                timeout=timeout,
                max_retries=max_retries,
                verbose=verbose,
                traceback_enabled=traceback_enabled,
            )
            registered_tools.append(tool_name)

        # Re-register the discovery tool to include all new agents
        self._register_agent_discovery_tool()

        logger.info(
            f"Added {len(agents)} agents as tools: {registered_tools}"
        )
        return registered_tools

    def _register_tool(
        self, tool_name: str, agent: AgentType
    ) -> None:
        """
        Register a single agent as an MCP tool.

        Args:
            tool_name: Name of the tool to register
            agent: The agent instance to register
        """
        config = self.tool_configs[tool_name]

        @self.mcp_server.tool(
            name=tool_name, description=config.tool_description
        )
        def agent_tool(
            task: str = None,
            img: str = None,
            imgs: List[str] = None,
            correct_answer: str = None,
            max_retries: int = None,
        ) -> Dict[str, Any]:
            """
            Execute the agent with the provided parameters.

            Args:
                task: The task or prompt to execute with this agent
                img: Optional image to be processed by the agent
                imgs: Optional list of images to be processed by the agent
                correct_answer: Optional correct answer for validation or comparison
                max_retries: Maximum number of retries (uses config default if None)

            Returns:
                Dict containing the agent's response and execution status
            """
            start_time = None
            if config.verbose:
                start_time = (
                    asyncio.get_event_loop().time()
                    if asyncio.get_event_loop().is_running()
                    else 0
                )
                logger.debug(
                    f"Starting execution of tool '{tool_name}' with task: {task[:100] if task else 'None'}..."
                )
                if img:
                    logger.debug(f"Processing single image: {img}")
                if imgs:
                    logger.debug(
                        f"Processing {len(imgs)} images: {imgs}"
                    )
                if correct_answer:
                    logger.debug(
                        f"Using correct answer for validation: {correct_answer[:50]}..."
                    )

            try:
                # Validate required parameters
                if not task:
                    error_msg = "No task provided"
                    logger.warning(
                        f"Tool '{tool_name}' called without task parameter"
                    )
                    return {
                        "result": "",
                        "success": False,
                        "error": error_msg,
                    }

                # Use queue-based execution if enabled
                if (
                    self.queue_enabled
                    and tool_name in self.task_queues
                ):
                    return self._execute_with_queue(
                        tool_name,
                        task,
                        img,
                        imgs,
                        correct_answer,
                        0,
                        max_retries,
                        True,
                        config,
                    )
                else:
                    # Fallback to direct execution
                    result = self._execute_agent_with_timeout(
                        agent,
                        task,
                        config.timeout,
                        img,
                        imgs,
                        correct_answer,
                    )

                    if config.verbose and start_time:
                        execution_time = (
                            asyncio.get_event_loop().time()
                            - start_time
                            if asyncio.get_event_loop().is_running()
                            else 0
                        )
                        logger.debug(
                            f"Tool '{tool_name}' completed successfully in {execution_time:.2f}s"
                        )

                    return {
                        "result": str(result),
                        "success": True,
                        "error": None,
                    }

            except Exception as e:
                error_msg = str(e)
                logger.error(
                    f"Error executing agent '{tool_name}': {error_msg}"
                )

                if config.traceback_enabled:
                    logger.error(f"Traceback for tool '{tool_name}':")
                    logger.error(traceback.format_exc())

                if config.verbose and start_time:
                    execution_time = (
                        asyncio.get_event_loop().time() - start_time
                        if asyncio.get_event_loop().is_running()
                        else 0
                    )
                    logger.debug(
                        f"Tool '{tool_name}' failed after {execution_time:.2f}s"
                    )

                return {
                    "result": "",
                    "success": False,
                    "error": error_msg,
                }

    def _execute_with_queue(
        self,
        tool_name: str,
        task: str,
        img: Optional[str],
        imgs: Optional[List[str]],
        correct_answer: Optional[str],
        priority: int,
        max_retries: Optional[int],
        wait_for_completion: bool,
        config: AgentToolConfig,
    ) -> Dict[str, Any]:
        """
        Execute a task using the queue system.

        Args:
            tool_name: Name of the tool/agent
            task: The task to execute
            img: Optional image to process
            imgs: Optional list of images to process
            correct_answer: Optional correct answer for validation
            priority: Task priority
            max_retries: Maximum number of retries
            wait_for_completion: Whether to wait for completion
            config: Tool configuration

        Returns:
            Dict containing the result or task information
        """
        try:
            # Use config max_retries if not specified
            if max_retries is None:
                max_retries = config.max_retries

            # Add task to queue
            task_id = self.task_queues[tool_name].add_task(
                task=task,
                img=img,
                imgs=imgs,
                correct_answer=correct_answer,
                priority=priority,
                max_retries=max_retries,
            )

            if not wait_for_completion:
                # Return task ID immediately
                return {
                    "task_id": task_id,
                    "status": "queued",
                    "success": True,
                    "message": f"Task '{task_id}' queued for agent '{tool_name}'",
                }

            # Wait for task completion
            return self._wait_for_task_completion(
                tool_name, task_id, config.timeout
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(
                f"Error adding task to queue for '{tool_name}': {error_msg}"
            )
            return {
                "result": "",
                "success": False,
                "error": error_msg,
            }

    def _wait_for_task_completion(
        self, tool_name: str, task_id: str, timeout: int
    ) -> Dict[str, Any]:
        """
        Wait for a task to complete.

        Args:
            tool_name: Name of the tool/agent
            task_id: ID of the task to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            Dict containing the task result
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            task = self.task_queues[tool_name].get_task(task_id)
            if not task:
                return {
                    "result": "",
                    "success": False,
                    "error": f"Task '{task_id}' not found",
                }

            if task.status == TaskStatus.COMPLETED:
                return {
                    "result": task.result or "",
                    "success": True,
                    "error": None,
                    "task_id": task_id,
                }
            elif task.status == TaskStatus.FAILED:
                return {
                    "result": "",
                    "success": False,
                    "error": task.error or "Task failed",
                    "task_id": task_id,
                }
            elif task.status == TaskStatus.CANCELLED:
                return {
                    "result": "",
                    "success": False,
                    "error": "Task was cancelled",
                    "task_id": task_id,
                }

            # Wait a bit before checking again
            time.sleep(0.1)

        # Timeout reached
        return {
            "result": "",
            "success": False,
            "error": f"Task '{task_id}' timed out after {timeout} seconds",
            "task_id": task_id,
        }

    def _execute_agent_with_timeout(
        self,
        agent: AgentType,
        task: str,
        timeout: int,
        img: str = None,
        imgs: List[str] = None,
        correct_answer: str = None,
    ) -> str:
        """
        Execute an agent with a timeout and all run method parameters.

        Args:
            agent: The agent to execute
            task: The task to execute
            timeout: Maximum execution time in seconds
            img: Optional image to be processed by the agent
            imgs: Optional list of images to be processed by the agent
            correct_answer: Optional correct answer for validation or comparison

        Returns:
            str: The agent's response

        Raises:
            TimeoutError: If execution exceeds timeout
            Exception: If agent execution fails
        """
        try:
            logger.debug(
                f"Executing agent '{agent.agent_name}' with timeout {timeout}s"
            )

            out = agent.run(
                task=task,
                img=img,
                imgs=imgs,
                correct_answer=correct_answer,
            )

            logger.debug(
                f"Agent '{agent.agent_name}' execution completed successfully"
            )
            return out

        except Exception as e:
            error_msg = f"Agent execution failed: {str(e)}"
            logger.error(
                f"Execution error for agent '{agent.agent_name}': {error_msg}"
            )
            if self.traceback_enabled:
                logger.error(
                    f"Traceback for agent '{agent.agent_name}':"
                )
                logger.error(traceback.format_exc())
            raise Exception(error_msg)

    def remove_agent(self, tool_name: str) -> bool:
        """
        Remove an agent from the MCP server.

        Args:
            tool_name: Name of the tool to remove

        Returns:
            bool: True if agent was removed, False if not found
        """
        if tool_name in self.agents:
            # Stop and remove task queue if it exists
            if tool_name in self.task_queues:
                self.task_queues[tool_name].stop_workers()
                del self.task_queues[tool_name]

            del self.agents[tool_name]
            del self.tool_configs[tool_name]
            logger.info(f"Removed agent tool '{tool_name}'")
            return True
        return False

    def list_agents(self) -> List[str]:
        """
        Get a list of all registered agent tool names.

        Returns:
            List[str]: List of tool names
        """
        agent_list = list(self.agents.keys())
        if self.verbose:
            logger.debug(
                f"Listing {len(agent_list)} registered agents: {agent_list}"
            )
        return agent_list

    def get_agent_info(
        self, tool_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific agent tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Dict containing agent information, or None if not found
        """
        if tool_name not in self.agents:
            if self.verbose:
                logger.debug(
                    f"Requested info for non-existent agent tool '{tool_name}'"
                )
            return None

        agent = self.agents[tool_name]
        config = self.tool_configs[tool_name]

        info = {
            "tool_name": tool_name,
            "agent_name": agent.agent_name,
            "agent_description": agent.agent_description,
            "model_name": getattr(agent, "model_name", "Unknown"),
            "max_loops": getattr(agent, "max_loops", 1),
            "tool_description": config.tool_description,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
            "verbose": config.verbose,
            "traceback_enabled": config.traceback_enabled,
        }

        if self.verbose:
            logger.debug(
                f"Retrieved info for agent tool '{tool_name}': {info}"
            )

        return info

    def get_queue_stats(
        self, tool_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get queue statistics for agents.

        Args:
            tool_name: Optional specific agent name. If None, returns stats for all agents.

        Returns:
            Dict containing queue statistics
        """
        if not self.queue_enabled:
            return {
                "success": False,
                "error": "Queue system is not enabled",
                "stats": {},
            }

        try:
            if tool_name:
                if tool_name not in self.task_queues:
                    return {
                        "success": False,
                        "error": f"Agent '{tool_name}' not found or has no queue",
                        "stats": {},
                    }

                stats = self.task_queues[tool_name].get_stats()
                return {
                    "success": True,
                    "agent_name": tool_name,
                    "stats": {
                        "total_tasks": stats.total_tasks,
                        "completed_tasks": stats.completed_tasks,
                        "failed_tasks": stats.failed_tasks,
                        "pending_tasks": stats.pending_tasks,
                        "processing_tasks": stats.processing_tasks,
                        "average_processing_time": stats.average_processing_time,
                        "queue_size": stats.queue_size,
                        "queue_status": self.task_queues[tool_name]
                        .get_status()
                        .value,
                    },
                }
            else:
                # Get stats for all agents
                all_stats = {}
                for name, queue in self.task_queues.items():
                    stats = queue.get_stats()
                    all_stats[name] = {
                        "total_tasks": stats.total_tasks,
                        "completed_tasks": stats.completed_tasks,
                        "failed_tasks": stats.failed_tasks,
                        "pending_tasks": stats.pending_tasks,
                        "processing_tasks": stats.processing_tasks,
                        "average_processing_time": stats.average_processing_time,
                        "queue_size": stats.queue_size,
                        "queue_status": queue.get_status().value,
                    }

                return {
                    "success": True,
                    "stats": all_stats,
                    "total_agents": len(all_stats),
                }

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error getting queue stats: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "stats": {},
            }

    def pause_agent_queue(self, tool_name: str) -> bool:
        """
        Pause the task queue for a specific agent.

        Args:
            tool_name: Name of the agent tool

        Returns:
            bool: True if paused successfully, False if not found
        """
        if not self.queue_enabled:
            logger.warning("Queue system is not enabled")
            return False

        if tool_name not in self.task_queues:
            logger.warning(
                f"Agent '{tool_name}' not found or has no queue"
            )
            return False

        try:
            self.task_queues[tool_name].pause_workers()
            logger.info(f"Paused queue for agent '{tool_name}'")
            return True
        except Exception as e:
            logger.error(
                f"Error pausing queue for agent '{tool_name}': {e}"
            )
            return False

    def resume_agent_queue(self, tool_name: str) -> bool:
        """
        Resume the task queue for a specific agent.

        Args:
            tool_name: Name of the agent tool

        Returns:
            bool: True if resumed successfully, False if not found
        """
        if not self.queue_enabled:
            logger.warning("Queue system is not enabled")
            return False

        if tool_name not in self.task_queues:
            logger.warning(
                f"Agent '{tool_name}' not found or has no queue"
            )
            return False

        try:
            self.task_queues[tool_name].resume_workers()
            logger.info(f"Resumed queue for agent '{tool_name}'")
            return True
        except Exception as e:
            logger.error(
                f"Error resuming queue for agent '{tool_name}': {e}"
            )
            return False

    def clear_agent_queue(self, tool_name: str) -> int:
        """
        Clear all pending tasks from an agent's queue.

        Args:
            tool_name: Name of the agent tool

        Returns:
            int: Number of tasks cleared, -1 if error
        """
        if not self.queue_enabled:
            logger.warning("Queue system is not enabled")
            return -1

        if tool_name not in self.task_queues:
            logger.warning(
                f"Agent '{tool_name}' not found or has no queue"
            )
            return -1

        try:
            cleared_count = self.task_queues[tool_name].clear_queue()
            logger.info(
                f"Cleared {cleared_count} tasks from queue for agent '{tool_name}'"
            )
            return cleared_count
        except Exception as e:
            logger.error(
                f"Error clearing queue for agent '{tool_name}': {e}"
            )
            return -1

    def get_task_status(
        self, tool_name: str, task_id: str
    ) -> Dict[str, Any]:
        """
        Get the status of a specific task.

        Args:
            tool_name: Name of the agent tool
            task_id: ID of the task

        Returns:
            Dict containing task status information
        """
        if not self.queue_enabled:
            return {
                "success": False,
                "error": "Queue system is not enabled",
                "task": None,
            }

        if tool_name not in self.task_queues:
            return {
                "success": False,
                "error": f"Agent '{tool_name}' not found or has no queue",
                "task": None,
            }

        try:
            task = self.task_queues[tool_name].get_task(task_id)
            if not task:
                return {
                    "success": False,
                    "error": f"Task '{task_id}' not found",
                    "task": None,
                }

            return {
                "success": True,
                "task": {
                    "task_id": task.task_id,
                    "status": task.status.value,
                    "created_at": task.created_at,
                    "result": task.result,
                    "error": task.error,
                    "retry_count": task.retry_count,
                    "max_retries": task.max_retries,
                    "priority": task.priority,
                },
            }
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            return {
                "success": False,
                "error": str(e),
                "task": None,
            }

    def cancel_task(self, tool_name: str, task_id: str) -> bool:
        """
        Cancel a specific task.

        Args:
            tool_name: Name of the agent tool
            task_id: ID of the task to cancel

        Returns:
            bool: True if cancelled successfully, False otherwise
        """
        if not self.queue_enabled:
            logger.warning("Queue system is not enabled")
            return False

        if tool_name not in self.task_queues:
            logger.warning(
                f"Agent '{tool_name}' not found or has no queue"
            )
            return False

        try:
            success = self.task_queues[tool_name].cancel_task(task_id)
            if success:
                logger.info(
                    f"Cancelled task '{task_id}' for agent '{tool_name}'"
                )
            else:
                logger.warning(
                    f"Could not cancel task '{task_id}' for agent '{tool_name}'"
                )
            return success
        except Exception as e:
            logger.error(f"Error cancelling task '{task_id}': {e}")
            return False

    def pause_all_queues(self) -> Dict[str, bool]:
        """
        Pause all agent queues.

        Returns:
            Dict mapping agent names to success status
        """
        if not self.queue_enabled:
            logger.warning("Queue system is not enabled")
            return {}

        results = {}
        for tool_name in self.task_queues.keys():
            results[tool_name] = self.pause_agent_queue(tool_name)

        logger.info(
            f"Paused {sum(results.values())} out of {len(results)} agent queues"
        )
        return results

    def resume_all_queues(self) -> Dict[str, bool]:
        """
        Resume all agent queues.

        Returns:
            Dict mapping agent names to success status
        """
        if not self.queue_enabled:
            logger.warning("Queue system is not enabled")
            return {}

        results = {}
        for tool_name in self.task_queues.keys():
            results[tool_name] = self.resume_agent_queue(tool_name)

        logger.info(
            f"Resumed {sum(results.values())} out of {len(results)} agent queues"
        )
        return results

    def clear_all_queues(self) -> Dict[str, int]:
        """
        Clear all agent queues.

        Returns:
            Dict mapping agent names to number of tasks cleared
        """
        if not self.queue_enabled:
            logger.warning("Queue system is not enabled")
            return {}

        results = {}
        total_cleared = 0
        for tool_name in self.task_queues.keys():
            cleared = self.clear_agent_queue(tool_name)
            results[tool_name] = cleared
            if cleared > 0:
                total_cleared += cleared

        logger.info(
            f"Cleared {total_cleared} tasks from all agent queues"
        )
        return results

    def _register_agent_discovery_tool(self) -> None:
        """
        Register the agent discovery tools that allow agents to learn about each other.
        """

        @self.mcp_server.tool(
            name="discover_agents",
            description="Discover information about other agents in the cluster including their name, description, system prompt (truncated to 200 chars), and tags.",
        )
        def discover_agents(agent_name: str = None) -> Dict[str, Any]:
            """
            Discover information about agents in the cluster.

            Args:
                agent_name: Optional specific agent name to get info for. If None, returns info for all agents.

            Returns:
                Dict containing agent information for discovery
            """
            try:
                if agent_name:
                    # Get specific agent info
                    if agent_name not in self.agents:
                        return {
                            "success": False,
                            "error": f"Agent '{agent_name}' not found",
                            "agents": [],
                        }

                    agent_info = self._get_agent_discovery_info(
                        agent_name
                    )
                    return {
                        "success": True,
                        "agents": [agent_info] if agent_info else [],
                    }
                else:
                    # Get all agents info
                    all_agents_info = []
                    for tool_name in self.agents.keys():
                        agent_info = self._get_agent_discovery_info(
                            tool_name
                        )
                        if agent_info:
                            all_agents_info.append(agent_info)

                    return {
                        "success": True,
                        "agents": all_agents_info,
                    }

            except Exception as e:
                error_msg = str(e)
                logger.error(
                    f"Error in discover_agents tool: {error_msg}"
                )
                return {
                    "success": False,
                    "error": error_msg,
                    "agents": [],
                }

        @self.mcp_server.tool(
            name="get_agent_details",
            description="Get detailed information about a single agent by name including configuration, capabilities, and metadata.",
        )
        def get_agent_details(agent_name: str) -> Dict[str, Any]:
            """
            Get detailed information about a specific agent.

            Args:
                agent_name: Name of the agent to get information for.

            Returns:
                Dict containing detailed agent information
            """
            try:
                if agent_name not in self.agents:
                    return {
                        "success": False,
                        "error": f"Agent '{agent_name}' not found",
                        "agent_info": None,
                    }

                agent_info = self.get_agent_info(agent_name)
                discovery_info = self._get_agent_discovery_info(
                    agent_name
                )

                return {
                    "success": True,
                    "agent_info": agent_info,
                    "discovery_info": discovery_info,
                }

            except Exception as e:
                error_msg = str(e)
                logger.error(
                    f"Error in get_agent_details tool: {error_msg}"
                )
                return {
                    "success": False,
                    "error": error_msg,
                    "agent_info": None,
                }

        @self.mcp_server.tool(
            name="get_agents_info",
            description="Get detailed information about multiple agents by providing a list of agent names.",
        )
        def get_agents_info(agent_names: List[str]) -> Dict[str, Any]:
            """
            Get detailed information about multiple agents.

            Args:
                agent_names: List of agent names to get information for.

            Returns:
                Dict containing detailed information for all requested agents
            """
            try:
                if not agent_names:
                    return {
                        "success": False,
                        "error": "No agent names provided",
                        "agents_info": [],
                    }

                agents_info = []
                not_found = []

                for agent_name in agent_names:
                    if agent_name in self.agents:
                        agent_info = self.get_agent_info(agent_name)
                        discovery_info = (
                            self._get_agent_discovery_info(agent_name)
                        )
                        agents_info.append(
                            {
                                "agent_name": agent_name,
                                "agent_info": agent_info,
                                "discovery_info": discovery_info,
                            }
                        )
                    else:
                        not_found.append(agent_name)

                return {
                    "success": True,
                    "agents_info": agents_info,
                    "not_found": not_found,
                    "total_found": len(agents_info),
                    "total_requested": len(agent_names),
                }

            except Exception as e:
                error_msg = str(e)
                logger.error(
                    f"Error in get_agents_info tool: {error_msg}"
                )
                return {
                    "success": False,
                    "error": error_msg,
                    "agents_info": [],
                }

        @self.mcp_server.tool(
            name="list_agents",
            description="Get a simple list of all available agent names in the cluster.",
        )
        def list_agents() -> Dict[str, Any]:
            """
            Get a list of all available agent names.

            Returns:
                Dict containing the list of agent names
            """
            try:
                agent_names = self.list_agents()
                return {
                    "success": True,
                    "agent_names": agent_names,
                    "total_count": len(agent_names),
                }

            except Exception as e:
                error_msg = str(e)
                logger.error(
                    f"Error in list_agents tool: {error_msg}"
                )
                return {
                    "success": False,
                    "error": error_msg,
                    "agent_names": [],
                }

        @self.mcp_server.tool(
            name="search_agents",
            description="Search for agents by name, description, tags, or capabilities using keyword matching.",
        )
        def search_agents(
            query: str, search_fields: List[str] = None
        ) -> Dict[str, Any]:
            """
            Search for agents using keyword matching.

            Args:
                query: Search query string
                search_fields: Optional list of fields to search in (name, description, tags, capabilities).
                              If None, searches all fields.

            Returns:
                Dict containing matching agents
            """
            try:
                if not query:
                    return {
                        "success": False,
                        "error": "No search query provided",
                        "matching_agents": [],
                    }

                # Default search fields
                if search_fields is None:
                    search_fields = [
                        "name",
                        "description",
                        "tags",
                        "capabilities",
                    ]

                query_lower = query.lower()
                matching_agents = []

                for tool_name in self.agents.keys():
                    discovery_info = self._get_agent_discovery_info(
                        tool_name
                    )
                    if not discovery_info:
                        continue

                    match_found = False

                    # Search in specified fields
                    for field in search_fields:
                        if (
                            field == "name"
                            and query_lower
                            in discovery_info.get(
                                "agent_name", ""
                            ).lower()
                        ):
                            match_found = True
                            break
                        elif (
                            field == "description"
                            and query_lower
                            in discovery_info.get(
                                "description", ""
                            ).lower()
                        ):
                            match_found = True
                            break
                        elif field == "tags":
                            tags = discovery_info.get("tags", [])
                            if any(
                                query_lower in tag.lower()
                                for tag in tags
                            ):
                                match_found = True
                                break
                        elif field == "capabilities":
                            capabilities = discovery_info.get(
                                "capabilities", []
                            )
                            if any(
                                query_lower in capability.lower()
                                for capability in capabilities
                            ):
                                match_found = True
                                break

                    if match_found:
                        matching_agents.append(discovery_info)

                return {
                    "success": True,
                    "matching_agents": matching_agents,
                    "total_matches": len(matching_agents),
                    "query": query,
                    "search_fields": search_fields,
                }

            except Exception as e:
                error_msg = str(e)
                logger.error(
                    f"Error in search_agents tool: {error_msg}"
                )
                return {
                    "success": False,
                    "error": error_msg,
                    "matching_agents": [],
                }

        @self.mcp_server.tool(
            name="get_server_info",
            description="Get comprehensive server information including metadata, configuration, tool details, queue stats, and network status.",
        )
        def get_server_info_tool() -> Dict[str, Any]:
            """
            Get comprehensive information about the MCP server and registered tools.

            Returns:
                Dict containing server information with the following fields:
                - server_name: Name of the server
                - description: Server description
                - total_tools/total_agents: Total number of agents registered
                - tools/agent_names: List of all agent names
                - created_at: Unix timestamp when server was created
                - created_at_iso: ISO formatted creation time
                - uptime_seconds: Server uptime in seconds
                - host: Server host address
                - port: Server port number
                - transport: Transport protocol used
                - log_level: Logging level
                - queue_enabled: Whether queue system is enabled
                - persistence_enabled: Whether persistence mode is enabled
                - network_monitoring_enabled: Whether network monitoring is enabled
                - persistence: Detailed persistence status
                - network: Detailed network status
                - tool_details: Detailed information about each agent tool
                - queue_config: Queue configuration (if queue enabled)
                - queue_stats: Queue statistics for each agent (if queue enabled)
            """
            try:
                server_info = self.get_server_info()
                return {
                    "success": True,
                    "server_info": server_info,
                }
            except Exception as e:
                error_msg = str(e)
                logger.error(
                    f"Error in get_server_info tool: {error_msg}"
                )
                return {
                    "success": False,
                    "error": error_msg,
                    "server_info": None,
                }

    def _register_queue_management_tools(self) -> None:
        """
        Register queue management tools for the MCP server.
        """

        @self.mcp_server.tool(
            name="get_queue_stats",
            description="Get queue statistics for agents including task counts, processing times, and queue status.",
        )
        def get_queue_stats(agent_name: str = None) -> Dict[str, Any]:
            """
            Get queue statistics for agents.

            Args:
                agent_name: Optional specific agent name. If None, returns stats for all agents.

            Returns:
                Dict containing queue statistics
            """
            return self.get_queue_stats(agent_name)

        @self.mcp_server.tool(
            name="pause_agent_queue",
            description="Pause the task queue for a specific agent.",
        )
        def pause_agent_queue(agent_name: str) -> Dict[str, Any]:
            """
            Pause the task queue for a specific agent.

            Args:
                agent_name: Name of the agent tool

            Returns:
                Dict containing success status
            """
            success = self.pause_agent_queue(agent_name)
            return {
                "success": success,
                "message": f"Queue for agent '{agent_name}' {'paused' if success else 'not found or already paused'}",
            }

        @self.mcp_server.tool(
            name="resume_agent_queue",
            description="Resume the task queue for a specific agent.",
        )
        def resume_agent_queue(agent_name: str) -> Dict[str, Any]:
            """
            Resume the task queue for a specific agent.

            Args:
                agent_name: Name of the agent tool

            Returns:
                Dict containing success status
            """
            success = self.resume_agent_queue(agent_name)
            return {
                "success": success,
                "message": f"Queue for agent '{agent_name}' {'resumed' if success else 'not found or already running'}",
            }

        @self.mcp_server.tool(
            name="clear_agent_queue",
            description="Clear all pending tasks from an agent's queue.",
        )
        def clear_agent_queue(agent_name: str) -> Dict[str, Any]:
            """
            Clear all pending tasks from an agent's queue.

            Args:
                agent_name: Name of the agent tool

            Returns:
                Dict containing number of tasks cleared
            """
            cleared_count = self.clear_agent_queue(agent_name)
            return {
                "success": cleared_count >= 0,
                "cleared_tasks": cleared_count,
                "message": (
                    f"Cleared {cleared_count} tasks from queue for agent '{agent_name}'"
                    if cleared_count >= 0
                    else f"Failed to clear queue for agent '{agent_name}'"
                ),
            }

        @self.mcp_server.tool(
            name="get_task_status",
            description="Get the status of a specific task by task ID.",
        )
        def get_task_status(
            agent_name: str, task_id: str
        ) -> Dict[str, Any]:
            """
            Get the status of a specific task.

            Args:
                agent_name: Name of the agent tool
                task_id: ID of the task

            Returns:
                Dict containing task status information
            """
            return self.get_task_status(agent_name, task_id)

        @self.mcp_server.tool(
            name="cancel_task",
            description="Cancel a specific task by task ID.",
        )
        def cancel_task(
            agent_name: str, task_id: str
        ) -> Dict[str, Any]:
            """
            Cancel a specific task.

            Args:
                agent_name: Name of the agent tool
                task_id: ID of the task to cancel

            Returns:
                Dict containing success status
            """
            success = self.cancel_task(agent_name, task_id)
            return {
                "success": success,
                "message": f"Task '{task_id}' {'cancelled' if success else 'not found or already processed'}",
            }

        @self.mcp_server.tool(
            name="pause_all_queues",
            description="Pause all agent queues.",
        )
        def pause_all_queues() -> Dict[str, Any]:
            """
            Pause all agent queues.

            Returns:
                Dict containing results for each agent
            """
            results = self.pause_all_queues()
            return {
                "success": True,
                "results": results,
                "total_agents": len(results),
                "successful_pauses": sum(results.values()),
            }

        @self.mcp_server.tool(
            name="resume_all_queues",
            description="Resume all agent queues.",
        )
        def resume_all_queues() -> Dict[str, Any]:
            """
            Resume all agent queues.

            Returns:
                Dict containing results for each agent
            """
            results = self.resume_all_queues()
            return {
                "success": True,
                "results": results,
                "total_agents": len(results),
                "successful_resumes": sum(results.values()),
            }

        @self.mcp_server.tool(
            name="clear_all_queues",
            description="Clear all agent queues.",
        )
        def clear_all_queues() -> Dict[str, Any]:
            """
            Clear all agent queues.

            Returns:
                Dict containing results for each agent
            """
            results = self.clear_all_queues()
            total_cleared = sum(results.values())
            return {
                "success": True,
                "results": results,
                "total_agents": len(results),
                "total_cleared": total_cleared,
            }

    def _get_agent_discovery_info(
        self, tool_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get discovery information for a specific agent.

        Args:
            tool_name: Name of the agent tool

        Returns:
            Dict containing agent discovery information, or None if not found
        """
        if tool_name not in self.agents:
            return None

        agent = self.agents[tool_name]

        # Get system prompt and truncate to 200 characters
        system_prompt = getattr(agent, "system_prompt", "")
        short_system_prompt = (
            system_prompt[:200] + "..."
            if len(system_prompt) > 200
            else system_prompt
        )

        # Get tags (if available)
        tags = getattr(agent, "tags", [])
        if not tags:
            tags = []

        # Get capabilities (if available)
        capabilities = getattr(agent, "capabilities", [])
        if not capabilities:
            capabilities = []

        # Get role (if available)
        role = getattr(agent, "role", "worker")

        # Get model name
        model_name = getattr(agent, "model_name", "Unknown")

        info = {
            "tool_name": tool_name,
            "agent_name": agent.agent_name,
            "description": agent.agent_description
            or "No description available",
            "short_system_prompt": short_system_prompt,
            "tags": tags,
            "capabilities": capabilities,
            "role": role,
            "model_name": model_name,
            "max_loops": getattr(agent, "max_loops", 1),
            "temperature": getattr(agent, "temperature", 0.5),
            "max_tokens": getattr(agent, "max_tokens", 4096),
        }

        if self.verbose:
            logger.debug(
                f"Retrieved discovery info for agent '{tool_name}': {info}"
            )

        return info

    def start_server(self) -> None:
        """
        Start the MCP server.

        Args:
            host: Host to bind the server to
            port: Port to bind the server to
        """
        logger.info(
            f"Starting MCP server '{self.server_name}' on {self.host}:{self.port}\n"
            f"Transport: {self.transport}\n"
            f"Log level: {self.log_level}\n"
            f"Verbose mode: {self.verbose}\n"
            f"Traceback enabled: {self.traceback_enabled}\n"
            f"Queue enabled: {self.queue_enabled}\n"
            f"Available tools: {self.list_agents()}"
        )

        if self.verbose:
            logger.debug(
                "Server configuration:\n"
                f"  - Server name: {self.server_name}\n"
                f"  - Host: {self.host}\n"
                f"  - Port: {self.port}\n"
                f"  - Transport: {self.transport}\n"
                f"  - Queue enabled: {self.queue_enabled}\n"
                f"  - Total agents: {len(self.agents)}"
            )
            for tool_name, config in self.tool_configs.items():
                logger.debug(
                    f"  - Tool '{tool_name}': timeout={config.timeout}s, verbose={config.verbose}, traceback={config.traceback_enabled}"
                )

            if self.queue_enabled:
                logger.debug(
                    f"  - Max workers per agent: {self.max_workers_per_agent}"
                )
                logger.debug(
                    f"  - Max queue size per agent: {self.max_queue_size_per_agent}"
                )
                logger.debug(
                    f"  - Processing timeout: {self.processing_timeout}s"
                )
                logger.debug(f"  - Retry delay: {self.retry_delay}s")

        try:
            self.mcp_server.run(transport=self.transport)
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        finally:
            # Clean up queues when server stops
            if self.queue_enabled:
                logger.info("Stopping all agent queues...")
                for tool_name in list(self.task_queues.keys()):
                    try:
                        self.task_queues[tool_name].stop_workers()
                        logger.debug(
                            f"Stopped queue for agent '{tool_name}'"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error stopping queue for agent '{tool_name}': {e}"
                        )

        logger.info(
            f"MCP Server '{self.server_name}' is ready with {len(self.agents)} tools"
        )
        logger.info(
            f"Tools available: {', '.join(self.list_agents())}"
        )

    def run(self) -> None:
        """
        Run the MCP server with optional persistence.

        If persistence is enabled, the server will automatically restart
        when stopped, up to max_restart_attempts times. This includes
        a failsafe mechanism to prevent infinite restart loops.
        """
        if not self._persistence_enabled:
            # Standard run without persistence
            self.start_server()
            return

        # Persistence-enabled run
        logger.info(
            f"Starting AOP server with persistence enabled (max restarts: {self.max_restart_attempts})"
        )

        while (
            not self._shutdown_requested
            and self._restart_count <= self.max_restart_attempts
        ):
            try:
                if self._restart_count > 0:
                    logger.info(
                        f"Restarting server (attempt {self._restart_count}/{self.max_restart_attempts})"
                    )
                    # Wait before restarting
                    time.sleep(self.restart_delay)

                # Reset restart count on successful start
                self._restart_count = 0
                self.start_server()

            except KeyboardInterrupt:
                if (
                    self._persistence_enabled
                    and not self._shutdown_requested
                ):
                    logger.warning(
                        "Server interrupted by user, but persistence is enabled. Restarting..."
                    )
                    self._restart_count += 1
                    continue
                else:
                    logger.info("Server shutdown requested by user")
                    break

            except Exception as e:
                if (
                    self._persistence_enabled
                    and not self._shutdown_requested
                ):
                    # Check if it's a network error
                    if self._is_network_error(e):
                        logger.warning(
                            "🌐 Network error detected, attempting reconnection..."
                        )
                        if self._handle_network_error(e):
                            # Network retry successful, continue with restart
                            self._restart_count += 1
                            continue
                        else:
                            # Network retry failed, give up
                            logger.critical(
                                "💀 Network reconnection failed permanently"
                            )
                            break
                    else:
                        # Non-network error, use standard restart logic
                        logger.error(
                            f"Server crashed with error: {e}"
                        )
                        self._restart_count += 1

                        if (
                            self._restart_count
                            > self.max_restart_attempts
                        ):
                            logger.critical(
                                f"Maximum restart attempts ({self.max_restart_attempts}) exceeded. Shutting down permanently."
                            )
                            break
                        else:
                            logger.info(
                                f"Will restart in {self.restart_delay} seconds..."
                            )
                            continue
                else:
                    # Check if it's a network error even without persistence
                    if self._is_network_error(e):
                        logger.error(
                            "🌐 Network error detected but persistence is disabled"
                        )
                        if self.network_monitoring:
                            logger.info(
                                "🔄 Attempting network reconnection..."
                            )
                            if self._handle_network_error(e):
                                # Try to start server again after network recovery
                                try:
                                    self.start_server()
                                    return
                                except Exception as retry_error:
                                    logger.error(
                                        f"Server failed after network recovery: {retry_error}"
                                    )
                                    raise
                            else:
                                logger.critical(
                                    "💀 Network reconnection failed"
                                )
                                raise
                        else:
                            logger.error(
                                "Network monitoring is disabled, cannot retry"
                            )
                            raise
                    else:
                        logger.error(
                            f"Server failed and persistence is disabled: {e}"
                        )
                        raise

        if self._restart_count > self.max_restart_attempts:
            logger.critical(
                "Server failed permanently due to exceeding maximum restart attempts"
            )
        elif self._shutdown_requested:
            logger.info("Server shutdown completed as requested")
        else:
            logger.info("Server stopped normally")

    def _is_network_error(self, error: Exception) -> bool:
        """
        Check if an error is network-related.

        Args:
            error: The exception to check

        Returns:
            bool: True if the error is network-related
        """
        network_errors = (
            ConnectionError,
            ConnectionRefusedError,
            ConnectionResetError,
            ConnectionAbortedError,
            TimeoutError,
            socket.gaierror,
            socket.timeout,
            OSError,
        )

        # Check if it's a direct network error
        if isinstance(error, network_errors):
            return True

        # Check error message for network-related keywords
        error_msg = str(error).lower()
        network_keywords = [
            "connection refused",
            "connection reset",
            "connection aborted",
            "network is unreachable",
            "no route to host",
            "timeout",
            "socket",
            "network",
            "connection",
            "refused",
            "reset",
            "aborted",
            "unreachable",
            "timeout",
        ]

        return any(
            keyword in error_msg for keyword in network_keywords
        )

    def _get_network_error_message(
        self, error: Exception, attempt: int
    ) -> str:
        """
        Get a custom error message for network-related errors.

        Args:
            error: The network error that occurred
            attempt: Current retry attempt number

        Returns:
            str: Custom error message
        """
        error_type = type(error).__name__
        error_msg = str(error)

        if isinstance(error, ConnectionRefusedError):
            return f"🌐 NETWORK ERROR: Connection refused to {self.host}:{self.port} (attempt {attempt}/{self.max_network_retries})"
        elif isinstance(error, ConnectionResetError):
            return f"🌐 NETWORK ERROR: Connection was reset by remote host (attempt {attempt}/{self.max_network_retries})"
        elif isinstance(error, ConnectionAbortedError):
            return f"🌐 NETWORK ERROR: Connection was aborted (attempt {attempt}/{self.max_network_retries})"
        elif isinstance(error, TimeoutError):
            return f"🌐 NETWORK ERROR: Connection timeout after {self.network_timeout}s (attempt {attempt}/{self.max_network_retries})"
        elif isinstance(error, socket.gaierror):
            return f"🌐 NETWORK ERROR: Host resolution failed for {self.host} (attempt {attempt}/{self.max_network_retries})"
        elif isinstance(error, OSError):
            return f"🌐 NETWORK ERROR: OS-level network error - {error_msg} (attempt {attempt}/{self.max_network_retries})"
        else:
            return f"🌐 NETWORK ERROR: {error_type} - {error_msg} (attempt {attempt}/{self.max_network_retries})"

    def _test_network_connectivity(self) -> bool:
        """
        Test network connectivity to the server host and port.

        Returns:
            bool: True if network is reachable, False otherwise
        """
        try:
            # Test if we can resolve the host
            socket.gethostbyname(self.host)

            # Test if we can connect to the port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.network_timeout)
            result = sock.connect_ex((self.host, self.port))
            sock.close()

            return result == 0
        except Exception as e:
            if self.verbose:
                logger.debug(f"Network connectivity test failed: {e}")
            return False

    def _handle_network_error(self, error: Exception) -> bool:
        """
        Handle network errors with retry logic.

        Args:
            error: The network error that occurred

        Returns:
            bool: True if should retry, False if should give up
        """
        if not self.network_monitoring:
            return False

        self._network_retry_count += 1
        self._last_network_error = error
        self._network_connected = False

        # Get custom error message
        error_msg = self._get_network_error_message(
            error, self._network_retry_count
        )
        logger.error(error_msg)

        # Check if we should retry
        if self._network_retry_count <= self.max_network_retries:
            logger.warning(
                f"🔄 Attempting to reconnect in {self.network_retry_delay} seconds..."
            )
            logger.info(
                f"📊 Network retry {self._network_retry_count}/{self.max_network_retries}"
            )

            # Wait before retry
            time.sleep(self.network_retry_delay)

            # Test connectivity before retry
            if self._test_network_connectivity():
                logger.info("✅ Network connectivity restored!")
                self._network_connected = True
                self._network_retry_count = (
                    0  # Reset on successful test
                )
                return True
            else:
                logger.warning(
                    "❌ Network connectivity test failed, will retry..."
                )
                return True
        else:
            logger.critical(
                f"💀 Maximum network retry attempts ({self.max_network_retries}) exceeded!"
            )
            logger.critical(
                "🚫 Giving up on network reconnection. Server will shut down."
            )
            return False

    def get_network_status(self) -> Dict[str, Any]:
        """
        Get current network status and statistics.

        Returns:
            Dict containing network status information
        """
        return {
            "network_monitoring_enabled": self.network_monitoring,
            "network_connected": self._network_connected,
            "network_retry_count": self._network_retry_count,
            "max_network_retries": self.max_network_retries,
            "network_retry_delay": self.network_retry_delay,
            "network_timeout": self.network_timeout,
            "last_network_error": (
                str(self._last_network_error)
                if self._last_network_error
                else None
            ),
            "remaining_network_retries": max(
                0,
                self.max_network_retries - self._network_retry_count,
            ),
            "host": self.host,
            "port": self.port,
        }

    def reset_network_retry_count(self) -> None:
        """
        Reset the network retry counter.

        This can be useful if you want to give the server a fresh
        set of network retry attempts.
        """
        self._network_retry_count = 0
        self._last_network_error = None
        self._network_connected = True
        logger.info("Network retry counter reset")

    def enable_persistence(self) -> None:
        """
        Enable persistence mode for the server.

        This allows the server to automatically restart when stopped,
        up to the maximum number of restart attempts.
        """
        self._persistence_enabled = True
        logger.info("Persistence mode enabled")

    def disable_persistence(self) -> None:
        """
        Disable persistence mode for the server.

        This will allow the server to shut down normally without
        automatic restarts.
        """
        self._persistence_enabled = False
        self._shutdown_requested = True
        logger.info(
            "Persistence mode disabled - server will shut down on next stop"
        )

    def request_shutdown(self) -> None:
        """
        Request a graceful shutdown of the server.

        If persistence is enabled, this will prevent automatic restarts
        and allow the server to shut down normally.
        """
        self._shutdown_requested = True
        logger.info(
            "Shutdown requested - server will stop after current operations complete"
        )

    def get_persistence_status(self) -> Dict[str, Any]:
        """
        Get the current persistence status and statistics.

        Returns:
            Dict containing persistence configuration and status
        """
        return {
            "persistence_enabled": self._persistence_enabled,
            "shutdown_requested": self._shutdown_requested,
            "restart_count": self._restart_count,
            "max_restart_attempts": self.max_restart_attempts,
            "restart_delay": self.restart_delay,
            "remaining_restarts": max(
                0, self.max_restart_attempts - self._restart_count
            ),
        }

    def reset_restart_count(self) -> None:
        """
        Reset the restart counter.

        This can be useful if you want to give the server a fresh
        set of restart attempts.
        """
        self._restart_count = 0
        logger.info("Restart counter reset")

    def get_server_info(self) -> Dict[str, Any]:
        """
        Get information about the MCP server and registered tools.

        Returns:
            Dict containing server information including metadata, configuration,
            and tool details
        """
        info = {
            "server_name": self.server_name,
            "description": self.description,
            "total_tools": len(self.agents),
            "total_agents": len(
                self.agents
            ),  # Alias for compatibility
            "tools": self.list_agents(),
            "agent_names": self.list_agents(),  # Alias for compatibility
            "created_at": self._created_at,
            "created_at_iso": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self._created_at)
            ),
            "uptime_seconds": time.time() - self._created_at,
            "verbose": self.verbose,
            "traceback_enabled": self.traceback_enabled,
            "log_level": self.log_level,
            "transport": self.transport,
            "host": self.host,
            "port": self.port,
            "queue_enabled": self.queue_enabled,
            "persistence_enabled": self._persistence_enabled,  # Top-level for compatibility
            "network_monitoring_enabled": self.network_monitoring,  # Top-level for compatibility
            "persistence": self.get_persistence_status(),
            "network": self.get_network_status(),
            "tool_details": {
                tool_name: self.get_agent_info(tool_name)
                for tool_name in self.agents.keys()
            },
        }

        # Add queue information if enabled
        if self.queue_enabled:
            info["queue_config"] = {
                "max_workers_per_agent": self.max_workers_per_agent,
                "max_queue_size_per_agent": self.max_queue_size_per_agent,
                "processing_timeout": self.processing_timeout,
                "retry_delay": self.retry_delay,
            }

            # Add queue stats for each agent
            queue_stats = {}
            for tool_name in self.agents.keys():
                if tool_name in self.task_queues:
                    stats = self.task_queues[tool_name].get_stats()
                    queue_stats[tool_name] = {
                        "status": self.task_queues[tool_name]
                        .get_status()
                        .value,
                        "total_tasks": stats.total_tasks,
                        "completed_tasks": stats.completed_tasks,
                        "failed_tasks": stats.failed_tasks,
                        "pending_tasks": stats.pending_tasks,
                        "processing_tasks": stats.processing_tasks,
                        "average_processing_time": stats.average_processing_time,
                        "queue_size": stats.queue_size,
                    }

            info["queue_stats"] = queue_stats

        if self.verbose:
            logger.debug(f"Retrieved server info: {info}")

        return info


class AOPCluster:
    """
    AOPCluster manages a cluster of MCP servers, allowing for the retrieval and searching
    of tools (agents) across multiple endpoints.

    Attributes:
        urls (List[str]): List of MCP server URLs to connect to.
        transport (str): The transport protocol to use (default: "streamable-http").
    """

    def __init__(
        self,
        urls: List[str],
        transport: str = "streamable-http",
        *args,
        **kwargs,
    ):
        """
        Initialize the AOPCluster.

        Args:
            urls (List[str]): List of MCP server URLs.
            transport (str, optional): Transport protocol to use. Defaults to "streamable-http".
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.urls = urls
        self.transport = transport

    def get_tools(
        self, output_type: Literal["json", "dict", "str"] = "dict"
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the list of tools (agents) from all MCP servers in the cluster.

        Args:
            output_type (Literal["json", "dict", "str"], optional): The format of the output.
                Can be "json", "dict", or "str". Defaults to "dict".

        Returns:
            List[Dict[str, Any]]: A list of tool information dictionaries.
        """
        return get_tools_for_multiple_mcp_servers(
            urls=self.urls,
            format="openai",
            output_type=output_type,
            transport=self.transport,
        )

    def find_tool_by_server_name(
        self, server_name: str
    ) -> Dict[str, Any]:
        """
        Find a tool by its server name (function name).

        Args:
            server_name (str): The name of the tool/function to find.

        Returns:
            Dict[str, Any]: Dictionary containing the tool information, or None if not found.
        """
        for tool in self.get_tools(output_type="dict"):
            if tool.get("function", {}).get("name") == server_name:
                return tool
        return None
