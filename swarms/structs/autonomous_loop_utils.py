"""
Autonomous Loop Utilities

This module contains prompts and tool definitions for the autonomous loop structure
used when max_loops="auto" and interactive=False.

The autonomous loop follows this structure:
1. Planning phase: Create a plan using create_plan tool
2. Execution phase: For each subtask, loop thinking -> tool actions -> observation
3. Summary phase: Generate comprehensive summary when all subtasks are complete

Available Tools:
- create_plan: Create a detailed plan with subtasks
- think: Analyze and decide next actions
- subtask_done: Mark a subtask as complete
- complete_task: Mark the main task as complete
- respond_to_user: Send messages to the user
- create_file: Create new files
- update_file: Update existing files
- read_file: Read file contents
- list_directory: List directory contents
- delete_file: Delete files
- run_bash: Execute bash/shell commands on the terminal
- create_sub_agent: Create specialized sub-agents for delegation
- assign_task: Assign tasks to sub-agents asynchronously
- get_task_status: Check status of background subagent tasks
- cancel_task: Cancel a running subagent task
"""

import os
import subprocess
import threading
import time
import uuid
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    wait,
    ALL_COMPLETED,
    FIRST_COMPLETED,
)
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from loguru import logger


# ============================================================================
# CONSTANTS
# ============================================================================


# Maximum iterations to prevent infinite loops
MAX_PLANNING_ATTEMPTS = 5
MAX_SUBTASK_ITERATIONS = 100
MAX_SUBTASK_LOOPS = 20
MAX_CONSECUTIVE_THINKS = 2


# ============================================================================
# SUBAGENT TASK TRACKING
# ============================================================================


class SubagentTaskStatus(str, Enum):
    """Status of a subagent task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SubagentTask:
    """Tracks a single subagent task execution."""

    id: str
    agent_ref: Any
    agent_id: str
    agent_name: str
    task_str: str
    status: SubagentTaskStatus = SubagentTaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    future: Optional[Future] = None
    parent_id: Optional[str] = None
    depth: int = 0
    retries: int = 0
    max_retries: int = 0
    retry_on: Optional[List[Type[Exception]]] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


class SubagentTaskRegistry:
    """
    Manages subagent tasks with ThreadPoolExecutor, status tracking,
    retry logic, depth-limited recursion, and result aggregation.
    """

    def __init__(
        self,
        max_depth: int = 3,
        max_workers: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self._tasks: Dict[str, SubagentTask] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers
        )
        self._lock = threading.Lock()

    def spawn(
        self,
        agent: Any,
        agent_id: str,
        task: str,
        parent_id: Optional[str] = None,
        depth: int = 0,
        max_retries: int = 0,
        retry_on: Optional[List[Type[Exception]]] = None,
        fail_fast: bool = True,
    ) -> str:
        """
        Spawn an agent task in the background. Returns task_id.
        Raises ValueError if depth > max_depth.
        """
        if depth > self.max_depth:
            raise ValueError(
                f"Subagent depth {depth} exceeds max_depth {self.max_depth}"
            )

        task_id = f"task-{uuid.uuid4().hex[:8]}"
        agent_name = getattr(agent, "agent_name", str(agent))
        st = SubagentTask(
            id=task_id,
            agent_ref=agent,
            agent_id=agent_id,
            agent_name=agent_name,
            task_str=task,
            parent_id=parent_id,
            depth=depth,
            max_retries=max_retries,
            retry_on=retry_on or [],
        )

        with self._lock:
            self._tasks[task_id] = st

        logger.info(
            f"[SubagentTaskRegistry] Spawned {task_id} | agent={agent_name} | depth={depth}"
        )

        st.status = SubagentTaskStatus.RUNNING
        future = self._executor.submit(
            self._execute_task, st, fail_fast
        )
        st.future = future
        return task_id

    def _execute_task(
        self, st: SubagentTask, fail_fast: bool
    ) -> Any:
        """Run the agent with retry logic."""
        agent_name = st.agent_name
        last_error = None

        for attempt in range(st.max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(
                        f"[SubagentTaskRegistry] Retry {attempt}/{st.max_retries} for {st.id}"
                    )
                    st.retries = attempt

                result = st.agent_ref.run(st.task_str)
                st.result = result
                st.status = SubagentTaskStatus.COMPLETED
                st.completed_at = time.time()
                logger.info(
                    f"[SubagentTaskRegistry] {st.id} completed | agent={agent_name} | "
                    f"duration={st.completed_at - st.created_at:.2f}s"
                )
                return result

            except Exception as e:
                last_error = e
                should_retry = attempt < st.max_retries and (
                    not st.retry_on
                    or any(
                        isinstance(e, exc_type)
                        for exc_type in st.retry_on
                    )
                )
                if should_retry:
                    continue

                st.error = e
                st.status = SubagentTaskStatus.FAILED
                st.completed_at = time.time()
                logger.error(
                    f"[SubagentTaskRegistry] {st.id} failed | agent={agent_name} | error={e}"
                )
                if fail_fast:
                    raise
                return None

        st.error = last_error
        st.status = SubagentTaskStatus.FAILED
        st.completed_at = time.time()
        if fail_fast:
            raise last_error
        return None

    def get_task(self, task_id: str) -> SubagentTask:
        """Get a task by ID."""
        if task_id not in self._tasks:
            raise KeyError(f"Task {task_id} not found")
        return self._tasks[task_id]

    def get_results(self) -> Dict[str, Any]:
        """Collect results from all completed/failed tasks."""
        results = {}
        for task_id, st in self._tasks.items():
            if st.status == SubagentTaskStatus.COMPLETED:
                results[task_id] = st.result
            elif st.status == SubagentTaskStatus.FAILED:
                results[task_id] = st.error
        return results

    def cancel(self, task_id: str) -> bool:
        """Cancel a task if it hasn't completed yet."""
        st = self.get_task(task_id)
        if st.future and st.future.cancel():
            st.status = SubagentTaskStatus.CANCELLED
            st.completed_at = time.time()
            logger.info(
                f"[SubagentTaskRegistry] {task_id} cancelled"
            )
            return True
        return False

    def gather(
        self,
        strategy: str = "wait_all",
        timeout: Optional[float] = None,
    ) -> List[Any]:
        """
        Wait for tasks and return results.

        Args:
            strategy: "wait_all" or "wait_first"
            timeout: Max seconds to wait
        """
        already_done = []
        pending_futures = {}
        for st in self._tasks.values():
            if st.status in (
                SubagentTaskStatus.COMPLETED,
                SubagentTaskStatus.FAILED,
            ):
                already_done.append(st)
            elif st.future is not None:
                pending_futures[st.future] = st

        if not pending_futures:
            return [
                st.error
                if st.status == SubagentTaskStatus.FAILED
                else st.result
                for st in already_done
            ]

        return_when = (
            FIRST_COMPLETED
            if strategy == "wait_first"
            else ALL_COMPLETED
        )
        done, _ = wait(
            pending_futures.keys(),
            timeout=timeout,
            return_when=return_when,
        )

        results = [
            st.error
            if st.status == SubagentTaskStatus.FAILED
            else st.result
            for st in already_done
        ]
        for future in done:
            try:
                result = future.result(timeout=0)
                results.append(result)
            except Exception as e:
                results.append(e)

        return results

    def shutdown(self):
        """Shut down the executor."""
        self._executor.shutdown(wait=False)
        logger.info("[SubagentTaskRegistry] Shut down")

    @property
    def tasks(self) -> Dict[str, SubagentTask]:
        return dict(self._tasks)


def _get_task_registry(agent: Any) -> SubagentTaskRegistry:
    """Lazy-init and return the task registry stored on agent."""
    if (
        not hasattr(agent, "_task_registry")
        or agent._task_registry is None
    ):
        max_depth = getattr(agent, "max_subagent_depth", 3)
        agent._task_registry = SubagentTaskRegistry(
            max_depth=max_depth
        )
    return agent._task_registry


# ============================================================================
# PROMPTS
# ============================================================================


def get_planning_prompt(task: str) -> str:
    """
    Get the planning phase prompt.

    Args:
        task: The task description

    Returns:
        str: Planning prompt
    """
    return f"""You need to create a comprehensive plan for the following task:

{task}

Use the create_plan tool to break down this task into manageable subtasks. Each subtask should be specific and actionable.
"""


def get_execution_prompt(
    subtask_id: str,
    subtask_desc: str,
    all_subtasks: List[Dict[str, Any]],
) -> str:
    """
    Get the execution phase prompt for a specific subtask.

    Args:
        subtask_id: The ID of the current subtask
        subtask_desc: The description of the current subtask
        all_subtasks: List of all subtasks with their status

    Returns:
        str: Execution prompt
    """
    subtask_status_list = "\n".join(
        [
            f"- {s['step_id']}: {s['status']} - {s['description']}"
            for s in all_subtasks
        ]
    )

    return f"""You are currently working on subtask: {subtask_id}
Description: {subtask_desc}

Current status of all subtasks:
{subtask_status_list}

Follow this workflow:
1. Use the 'think' tool to analyze what needs to be done (optional, but recommended)
2. Use available tools to complete the work
3. When the subtask is complete, use 'subtask_done' to mark it as finished

Remember: Only call subtask_done when the work is ACTUALLY DONE, not when you're about to start.
"""


def get_summary_prompt() -> str:
    """
    Get the final summary phase prompt.

    Returns:
        str: Summary prompt
    """
    return (
        "All subtasks are complete.\n"
        "Generate a clear, comprehensive final summary using the `complete_task` tool.\n"
        "Your summary should:\n"
        "- Restate the original task and confirm completion.\n"
        "- Outline major accomplishments and deliverables.\n"
        "- Present key results and findings (including important data or insights).\n"
        "- Briefly summarize each subtask's status (including any failures).\n"
        "- Capture notable lessons learned, challenges, and future recommendations.\n"
        "Clearly state the task's overall success or any limitations.\n"
        "Use the `complete_task` tool with: task_id, summary, success (true/false), results (optional), and lessons_learned (optional).\n"
        "Be concise, well-organized, and thorough—this is the user's final deliverable."
    )


# ============================================================================
# TOOL SCHEMAS
# ============================================================================


def get_autonomous_planning_tools() -> List[Dict[str, Any]]:
    """
    Get tool definitions for autonomous planning and execution.

    Returns:
        List of tool definition dictionaries
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "create_plan",
                "description": "Create a detailed plan for completing a task",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_description": {
                            "type": "string",
                            "description": "Description of the task to be completed",
                        },
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "step_id": {"type": "string"},
                                    "description": {"type": "string"},
                                    "priority": {
                                        "type": "string",
                                        "enum": [
                                            "low",
                                            "medium",
                                            "high",
                                            "critical",
                                        ],
                                    },
                                    "dependencies": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": [
                                    "step_id",
                                    "description",
                                    "priority",
                                ],
                            },
                        },
                    },
                    "required": ["task_description", "steps"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "think",
                "description": "Analyze current situation and decide next actions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "current_state": {
                            "type": "string",
                            "description": "Current state of the task execution",
                        },
                        "analysis": {
                            "type": "string",
                            "description": "Analysis of the current situation",
                        },
                        "next_actions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of next actions to take",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Confidence level in the analysis (0-1)",
                        },
                    },
                    "required": [
                        "current_state",
                        "analysis",
                        "next_actions",
                        "confidence",
                    ],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "subtask_done",
                "description": "Mark a subtask as completed and move to the next task in the plan",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "The ID of the completed subtask",
                        },
                        "summary": {
                            "type": "string",
                            "description": "Summary of what was accomplished",
                        },
                        "success": {
                            "type": "boolean",
                            "description": "Whether the subtask was completed successfully",
                        },
                    },
                    "required": ["task_id", "summary", "success"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "complete_task",
                "description": "Mark the main task as complete and provide comprehensive summary",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "The ID of the main task",
                        },
                        "summary": {
                            "type": "string",
                            "description": "Comprehensive summary of the entire task completion",
                        },
                        "success": {
                            "type": "boolean",
                            "description": "Whether the main task was successful",
                        },
                        "results": {
                            "type": "string",
                            "description": "Detailed results (optional)",
                        },
                        "lessons_learned": {
                            "type": "string",
                            "description": "Key insights (optional)",
                        },
                    },
                    "required": ["task_id", "summary", "success"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "respond_to_user",
                "description": "Send a message or response to the user. Use this when you need to communicate important information, ask questions, or provide updates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message to send to the user",
                        },
                        "message_type": {
                            "type": "string",
                            "enum": [
                                "info",
                                "question",
                                "warning",
                                "error",
                                "success",
                            ],
                            "description": "Type of message (default: info)",
                        },
                    },
                    "required": ["message"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_file",
                "description": "Create a new file with the specified content. The file will be created in the agent's workspace directory if a relative path is provided.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to create (relative to workspace or absolute)",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file",
                        },
                    },
                    "required": ["file_path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_file",
                "description": "Update an existing file with new content. You can replace the entire file or append to it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to update (relative to workspace or absolute)",
                        },
                        "content": {
                            "type": "string",
                            "description": "New content to write to the file",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["replace", "append"],
                            "description": "Update mode: 'replace' to overwrite file, 'append' to add to end (default: replace)",
                        },
                    },
                    "required": ["file_path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file. Returns the file content as a string.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to read (relative to workspace or absolute)",
                        },
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List files and directories in a specified path. Returns a list of items in the directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory_path": {
                            "type": "string",
                            "description": "Path to the directory to list (relative to workspace or absolute). If empty, lists workspace root.",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_file",
                "description": "Delete a file. Use with caution - this action cannot be undone.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to delete (relative to workspace or absolute)",
                        },
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_bash",
                "description": "Execute a bash/shell command on the terminal. Use this to run system commands, scripts, or any shell operations. Returns stdout and stderr.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash/shell command to execute (e.g. 'ls -la', 'python script.py')",
                        },
                        "timeout_seconds": {
                            "type": "integer",
                            "description": "Maximum seconds to wait for the command (default: 60). Use to avoid hanging on long-running commands.",
                        },
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_sub_agent",
                "description": "Create one or more sub-agents that can work on specialized tasks. Sub-agents are cached and can be reused across different task assignments.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agents": {
                            "type": "array",
                            "description": "List of sub-agents to create",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "agent_name": {
                                        "type": "string",
                                        "description": "Name of the sub-agent",
                                    },
                                    "agent_description": {
                                        "type": "string",
                                        "description": "Description of the sub-agent's role and capabilities",
                                    },
                                    "system_prompt": {
                                        "type": "string",
                                        "description": "Custom system prompt for the sub-agent (optional). If not provided, a default prompt based on the agent description will be used.",
                                    },
                                },
                                "required": [
                                    "agent_name",
                                    "agent_description",
                                ],
                            },
                        },
                    },
                    "required": ["agents"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "assign_task",
                "description": "Assign tasks to one or more sub-agents. Tasks run concurrently in background threads. Supports retry, timeout, and different completion strategies.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "assignments": {
                            "type": "array",
                            "description": "List of task assignments for sub-agents",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "agent_id": {
                                        "type": "string",
                                        "description": "ID of the sub-agent to assign the task to",
                                    },
                                    "task": {
                                        "type": "string",
                                        "description": "The task description for the sub-agent",
                                    },
                                    "task_id": {
                                        "type": "string",
                                        "description": "Unique identifier for this task assignment (optional)",
                                    },
                                },
                                "required": ["agent_id", "task"],
                            },
                        },
                        "wait_for_completion": {
                            "type": "boolean",
                            "description": "Whether to wait for tasks to complete before returning results (default: true). If false, tasks run in background and you can check status later with get_task_status.",
                        },
                        "strategy": {
                            "type": "string",
                            "enum": [
                                "wait_all",
                                "wait_first",
                            ],
                            "description": "Completion strategy: 'wait_all' waits for all tasks (default), 'wait_first' returns as soon as any task completes.",
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Maximum seconds to wait for task completion. If exceeded, returns whatever results are available.",
                        },
                        "max_retries": {
                            "type": "integer",
                            "description": "Maximum number of retry attempts per task on failure (default: 0, no retries).",
                        },
                        "fail_fast": {
                            "type": "boolean",
                            "description": "If true (default), failed tasks raise errors. If false, failed tasks return None and other tasks continue.",
                        },
                    },
                    "required": ["assignments"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_task_status",
                "description": "Check the status of background subagent tasks. Use this after dispatching tasks with wait_for_completion=false.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific task IDs to check (optional, defaults to all tasks).",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cancel_task",
                "description": "Cancel a running background subagent task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "ID of the task to cancel.",
                        },
                    },
                    "required": ["task_id"],
                },
            },
        },
    ]


def get_autonomous_loop_tool_names() -> List[str]:
    """
    Return a list of all autonomous loop tool names.

    Returns:
        List of tool name strings (e.g. ["create_plan", "think", "subtask_done", ...])
    """
    return [
        t["function"]["name"]
        for t in get_autonomous_planning_tools()
        if t.get("type") == "function" and "function" in t
    ]


# ============================================================================
# TOOL HANDLERS
# ============================================================================


def respond_to_user_tool(
    agent: Any, message: str, message_type: str = "info", **kwargs
) -> str:
    """
    Send a message to the user.

    Args:
        agent: The agent instance
        message: The message to send
        message_type: Type of message (info, question, warning, error, success)
        **kwargs: Additional arguments

    Returns:
        str: Confirmation message
    """
    if agent.print_on:
        from swarms.utils.formatter import formatter

        formatter.print_panel(
            message,
            title=f"Message to User [{message_type.upper()}]",
        )

    # Add to memory
    agent.short_memory.add(
        role="User Communication",
        content=f"[{message_type.upper()}] {message}",
    )

    if agent.verbose:
        logger.info(
            f"Sent message to user ({message_type}): {message}"
        )

    return f"Message sent to user: {message}"


def create_file_tool(
    agent: Any, file_path: str, content: str, **kwargs
) -> str:
    """
    Create a new file with content.

    Args:
        agent: The agent instance
        file_path: Path to the file (relative to workspace or absolute)
        content: Content to write to the file
        **kwargs: Additional arguments

    Returns:
        str: Path to the created file or error message
    """
    try:
        # Resolve path - if relative, use agent workspace
        if not os.path.isabs(file_path):
            workspace_dir = agent._get_agent_workspace_dir()
            full_path = os.path.join(workspace_dir, file_path)
        else:
            full_path = file_path

        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # Check if file already exists
        if os.path.exists(full_path):
            return f"Error: File already exists at {full_path}. Use update_file to modify existing files."

        # Write file
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Add to memory
        agent.short_memory.add(
            role="File Operations",
            content=f"Created file: {full_path}",
        )

        if agent.verbose:
            logger.info(f"Created file: {full_path}")

        return f"Successfully created file: {full_path}"
    except Exception as e:
        error_msg = f"Error creating file {file_path}: {str(e)}"
        logger.error(error_msg)
        agent.short_memory.add(
            role="File Operations",
            content=f"Error: {error_msg}",
        )
        return error_msg


def update_file_tool(
    agent: Any,
    file_path: str,
    content: str,
    mode: str = "replace",
    **kwargs,
) -> str:
    """
    Update an existing file with new content.

    Args:
        agent: The agent instance
        file_path: Path to the file (relative to workspace or absolute)
        content: New content to write
        mode: 'replace' to overwrite, 'append' to add to end
        **kwargs: Additional arguments

    Returns:
        str: Success message or error message
    """
    try:
        # Resolve path - if relative, use agent workspace
        if not os.path.isabs(file_path):
            workspace_dir = agent._get_agent_workspace_dir()
            full_path = os.path.join(workspace_dir, file_path)
        else:
            full_path = file_path

        # Check if file exists
        if not os.path.exists(full_path):
            return f"Error: File does not exist at {full_path}. Use create_file to create new files."

        # Update file based on mode
        if mode == "append":
            with open(full_path, "a", encoding="utf-8") as f:
                f.write(content)
            action = "appended to"
        else:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            action = "updated"

        # Add to memory
        agent.short_memory.add(
            role="File Operations",
            content=f"{action.capitalize()} file: {full_path}",
        )

        if agent.verbose:
            logger.info(f"{action.capitalize()} file: {full_path}")

        return f"Successfully {action} file: {full_path}"
    except Exception as e:
        error_msg = f"Error updating file {file_path}: {str(e)}"
        logger.error(error_msg)
        agent.short_memory.add(
            role="File Operations",
            content=f"Error: {error_msg}",
        )
        return error_msg


def read_file_tool(agent: Any, file_path: str, **kwargs) -> str:
    """
    Read the contents of a file.

    Args:
        agent: The agent instance
        file_path: Path to the file (relative to workspace or absolute)
        **kwargs: Additional arguments

    Returns:
        str: File contents or error message
    """
    try:
        # Resolve path - if relative, use agent workspace
        if not os.path.isabs(file_path):
            workspace_dir = agent._get_agent_workspace_dir()
            full_path = os.path.join(workspace_dir, file_path)
        else:
            full_path = file_path

        # Check if file exists
        if not os.path.exists(full_path):
            return f"Error: File does not exist at {full_path}"

        # Read file
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Add to memory
        agent.short_memory.add(
            role="File Operations",
            content=f"Read file: {full_path} ({len(content)} characters)",
        )

        if agent.verbose:
            logger.info(f"Read file: {full_path}")

        return content
    except Exception as e:
        error_msg = f"Error reading file {file_path}: {str(e)}"
        logger.error(error_msg)
        agent.short_memory.add(
            role="File Operations",
            content=f"Error: {error_msg}",
        )
        return error_msg


def list_directory_tool(
    agent: Any, directory_path: str = "", **kwargs
) -> str:
    """
    List files and directories in a specified path.

    Args:
        agent: The agent instance
        directory_path: Path to the directory (relative to workspace or absolute). Empty string for workspace root.
        **kwargs: Additional arguments

    Returns:
        str: Formatted list of directory contents
    """
    try:
        # Resolve path - if relative or empty, use agent workspace
        if not directory_path or not os.path.isabs(directory_path):
            workspace_dir = agent._get_agent_workspace_dir()
            if directory_path:
                full_path = os.path.join(
                    workspace_dir, directory_path
                )
            else:
                full_path = workspace_dir
        else:
            full_path = directory_path

        # Check if directory exists
        if not os.path.exists(full_path):
            return f"Error: Directory does not exist at {full_path}"

        if not os.path.isdir(full_path):
            return f"Error: Path is not a directory: {full_path}"

        # List directory contents
        items = []
        for item in sorted(os.listdir(full_path)):
            item_path = os.path.join(full_path, item)
            item_type = "DIR" if os.path.isdir(item_path) else "FILE"
            size = (
                os.path.getsize(item_path)
                if os.path.isfile(item_path)
                else 0
            )
            items.append(f"{item_type:4} {item:50} {size:>10} bytes")

        result = f"Directory listing for: {full_path}\n\n"
        result += "\n".join(items) if items else "Directory is empty"

        # Add to memory
        agent.short_memory.add(
            role="File Operations",
            content=f"Listed directory: {full_path} ({len(items)} items)",
        )

        if agent.verbose:
            logger.info(f"Listed directory: {full_path}")

        return result
    except Exception as e:
        error_msg = (
            f"Error listing directory {directory_path}: {str(e)}"
        )
        logger.error(error_msg)
        agent.short_memory.add(
            role="File Operations",
            content=f"Error: {error_msg}",
        )
        return error_msg


def delete_file_tool(agent: Any, file_path: str, **kwargs) -> str:
    """
    Delete a file.

    Args:
        agent: The agent instance
        file_path: Path to the file (relative to workspace or absolute)
        **kwargs: Additional arguments

    Returns:
        str: Success message or error message
    """
    try:
        # Resolve path - if relative, use agent workspace
        if not os.path.isabs(file_path):
            workspace_dir = agent._get_agent_workspace_dir()
            full_path = os.path.join(workspace_dir, file_path)
        else:
            full_path = file_path

        # Check if file exists
        if not os.path.exists(full_path):
            return f"Error: File does not exist at {full_path}"

        # Safety check - don't delete directories
        if os.path.isdir(full_path):
            return (
                f"Error: Path is a directory, not a file: {full_path}"
            )

        # Delete file
        os.remove(full_path)

        # Add to memory
        agent.short_memory.add(
            role="File Operations",
            content=f"Deleted file: {full_path}",
        )

        if agent.verbose:
            logger.info(f"Deleted file: {full_path}")

        return f"Successfully deleted file: {full_path}"
    except Exception as e:
        error_msg = f"Error deleting file {file_path}: {str(e)}"
        logger.error(error_msg)
        agent.short_memory.add(
            role="File Operations",
            content=f"Error: {error_msg}",
        )
        return error_msg


_BASH_BLOCKLIST = [
    # Recursive/forced deletion
    ("rm", "-rf"),
    ("rm", "-fr"),
    ("rm", "-r"),
    # Pipe command output into a shell interpreter
    ("| sh",),
    ("| bash",),
    ("| zsh",),
    ("| python",),
    ("| python3",),
    ("| perl",),
    ("| ruby",),
    # Raw disk writes
    ("dd", "if="),
    ("mkfs",),
    ("> /dev/sd",),
    ("> /dev/nvme",),
    # Fork bomb pattern
    (":(){",),
    # System shutdown / reboot
    ("shutdown",),
    ("reboot",),
    ("halt",),
    ("poweroff",),
    # Privilege escalation helpers writing to sensitive paths
    ("chmod 777 /",),
    ("chown", "/etc"),
    ("chown", "/bin"),
]

_BASH_MAX_LENGTH = 512


def _check_bash_command(command: str) -> str | None:
    """Return a rejection reason if *command* matches a dangerous pattern, else None."""
    if len(command) > _BASH_MAX_LENGTH:
        return f"Command exceeds maximum allowed length of {_BASH_MAX_LENGTH} characters."
    cmd_lower = command.lower()
    for pattern in _BASH_BLOCKLIST:
        if all(token in cmd_lower for token in pattern):
            return f"Command blocked: matches dangerous pattern {pattern!r}."
    return None


def run_bash_tool(
    agent: Any, command: str, timeout_seconds: int = 60, **kwargs
) -> str:
    """
    Execute a bash/shell command on the terminal.

    Args:
        agent: The agent instance
        command: The bash/shell command to execute
        timeout_seconds: Maximum seconds to wait (default: 60)
        **kwargs: Additional arguments

    Returns:
        str: Command stdout and stderr, or error message
    """
    # --- Security: reject commands that match known-dangerous patterns --------
    rejection = _check_bash_command(command)
    if rejection:
        logger.warning(
            f"run_bash_tool blocked command: {command!r} — {rejection}"
        )
        agent.short_memory.add(
            role="Terminal",
            content=f"Blocked (security): {command[:100]}{'...' if len(command) > 100 else ''}",
        )
        return f"Error: {rejection}"
    # -------------------------------------------------------------------------

    try:
        # Run in process cwd (where the user started the script) so commands like
        # ls -la and python script.py see the project directory, not the agent workspace.
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=None,  # use process current working directory
            encoding="utf-8",
            errors="replace",
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        output_parts = []
        if stdout:
            output_parts.append(f"stdout:\n{stdout}")
        if stderr:
            output_parts.append(f"stderr:\n{stderr}")
        if not output_parts:
            output_parts.append("(no output)")

        result_msg = (
            f"Command exited with code {result.returncode}\n"
            + "\n".join(output_parts)
        )

        # Add to memory
        agent.short_memory.add(
            role="Terminal",
            content=f"Executed: {command[:100]}{'...' if len(command) > 100 else ''} -> exit {result.returncode}",
        )

        if agent.verbose:
            logger.info(f"Executed bash command: {command[:80]}...")

        return result_msg.strip()
    except subprocess.TimeoutExpired:
        error_msg = f"Error: Command timed out after {timeout_seconds} seconds"
        logger.error(error_msg)
        agent.short_memory.add(
            role="Terminal",
            content=f"Timeout: {command[:80]}...",
        )
        return error_msg
    except Exception as e:
        error_msg = f"Error executing command: {str(e)}"
        logger.error(error_msg)
        agent.short_memory.add(
            role="Terminal",
            content=f"Error: {error_msg}",
        )
        return error_msg


def create_sub_agent_tool(
    agent: Any, agents: List[Dict[str, str]], **kwargs
) -> str:
    """
    Create one or more sub-agents that can work on specialized tasks.
    Sub-agents are cached in the main agent's sub_agents dictionary.
    Enforces depth limits to prevent runaway recursion.

    Args:
        agent: The main agent instance
        agents: List of agent specifications with agent_name, agent_description, and optional system_prompt
        **kwargs: Additional arguments

    Returns:
        str: Success message with created agent IDs or error message
    """
    try:
        # Initialize sub_agents dict if it doesn't exist
        if not hasattr(agent, "sub_agents"):
            agent.sub_agents = {}

        # Depth tracking
        parent_depth = getattr(agent, "_subagent_depth", 0)
        max_depth = getattr(agent, "max_subagent_depth", 3)
        if parent_depth >= max_depth:
            return (
                f"Error: Maximum subagent depth ({max_depth}) reached. "
                f"Cannot create deeper sub-agents."
            )

        created_agents = []

        for agent_spec in agents:
            agent_name = agent_spec.get("agent_name")
            agent_description = agent_spec.get("agent_description")
            system_prompt = agent_spec.get("system_prompt")

            if not agent_name or not agent_description:
                return "Error: Each agent must have agent_name and agent_description"

            # Default system_prompt when not provided or empty
            if not system_prompt:
                system_prompt = (
                    f"You are {agent_name}, a specialized assistant. "
                    f"{agent_description}"
                )

            agent_id = f"sub-agent-{uuid.uuid4().hex[:8]}"

            # Import Agent class to create sub-agent (deferred to avoid circular import)
            from swarms.structs.agent import Agent

            # Create sub-agent with the same LLM as parent
            sub_agent = Agent(
                id=agent_id,
                agent_name=agent_name,
                agent_description=agent_description,
                system_prompt=system_prompt,
                model_name=agent.model_name,
                max_loops=1,
                max_subagent_depth=max_depth,
                print_on=False,
            )

            # Propagate depth tracking
            sub_agent._subagent_depth = parent_depth + 1

            # Cache the sub-agent
            agent.sub_agents[agent_id] = {
                "agent": sub_agent,
                "name": agent_name,
                "description": agent_description,
                "system_prompt": system_prompt,
                "depth": parent_depth + 1,
                "parent_agent_id": getattr(agent, "id", None),
                "created_at": str(
                    __import__("datetime").datetime.now()
                ),
            }

            created_agents.append(
                f"{agent_name} (ID: {agent_id})"
            )

            logger.info(
                f"[SubagentTaskRegistry] Created sub-agent: {agent_name} "
                f"(ID: {agent_id}, depth: {parent_depth + 1})"
            )

        # Add to memory
        result_msg = (
            f"Successfully created {len(created_agents)} sub-agent(s): "
            + ", ".join(created_agents)
        )
        agent.short_memory.add(
            role="Sub-Agent Management",
            content=result_msg,
        )

        return result_msg

    except Exception as e:
        error_msg = f"Error creating sub-agents: {str(e)}"
        logger.error(error_msg)
        agent.short_memory.add(
            role="Sub-Agent Management",
            content=f"Error: {error_msg}",
        )
        return error_msg


def assign_task_tool(
    agent: Any,
    assignments: List[Dict[str, str]],
    wait_for_completion: bool = True,
    strategy: str = "wait_all",
    timeout: Optional[float] = None,
    max_retries: int = 0,
    fail_fast: bool = True,
    **kwargs,
) -> str:
    """
    Assign tasks to one or more sub-agents. Tasks run concurrently
    via ThreadPoolExecutor with retry logic, timeout, and aggregation strategies.

    Args:
        agent: The main agent instance
        assignments: List of task assignments with agent_id and task
        wait_for_completion: Whether to wait for tasks to complete
        strategy: "wait_all" (default) or "wait_first"
        timeout: Max seconds to wait for completion
        max_retries: Max retry attempts per task on failure (default: 0)
        fail_fast: If True, failed tasks raise errors. If False, failures are captured.
        **kwargs: Additional arguments

    Returns:
        str: Results from all sub-agents or error message
    """
    try:
        # Check if sub_agents exist
        if not hasattr(agent, "sub_agents") or not agent.sub_agents:
            return "Error: No sub-agents have been created. Use create_sub_agent first."

        # Validate all agent IDs exist
        for assignment in assignments:
            agent_id = assignment.get("agent_id")
            if agent_id not in agent.sub_agents:
                return (
                    f"Error: Sub-agent with ID '{agent_id}' not found. "
                    f"Available agents: {list(agent.sub_agents.keys())}"
                )

        registry = _get_task_registry(agent)
        task_ids = []

        for idx, assignment in enumerate(assignments):
            agent_id = assignment.get("agent_id")
            task_str = assignment.get("task")
            task_id_hint = assignment.get(
                "task_id", f"task-{idx + 1}"
            )

            sub_agent_data = agent.sub_agents[agent_id]
            sub_agent = sub_agent_data["agent"]
            depth = sub_agent_data.get("depth", 0)

            spawned_id = registry.spawn(
                agent=sub_agent,
                agent_id=agent_id,
                task=task_str,
                parent_id=getattr(agent, "id", None),
                depth=depth,
                max_retries=max_retries,
                fail_fast=fail_fast,
            )
            task_ids.append(
                (spawned_id, sub_agent_data["name"], task_id_hint)
            )

        if not wait_for_completion:
            # Non-blocking: return task IDs for later retrieval
            id_list = ", ".join(
                f"{name} ({tid})" for tid, name, _ in task_ids
            )
            msg = f"Dispatched {len(task_ids)} task(s) in background: {id_list}. Use get_task_status to check progress."
            agent.short_memory.add(
                role="Sub-Agent Execution", content=msg
            )
            return msg

        # Blocking: wait for results
        registry.gather(strategy=strategy, timeout=timeout)

        # Format results
        result_lines = [
            f"Completed {len(task_ids)} task assignment(s):\n"
        ]
        for tid, name, hint in task_ids:
            task_obj = registry.get_task(tid)
            if task_obj.status == SubagentTaskStatus.COMPLETED:
                result_lines.append(
                    f"\n[{name}] Task {hint} (ID: {tid}):"
                )
                result_lines.append(f"Status: completed")
                result_lines.append(
                    f"Result: {task_obj.result}\n"
                )
            elif task_obj.status == SubagentTaskStatus.FAILED:
                result_lines.append(
                    f"\n[{name}] Task {hint} (ID: {tid}) FAILED:"
                )
                result_lines.append(f"Error: {task_obj.error}")
                result_lines.append(
                    f"Retries: {task_obj.retries}/{task_obj.max_retries}\n"
                )
            else:
                result_lines.append(
                    f"\n[{name}] Task {hint} (ID: {tid}):"
                )
                result_lines.append(
                    f"Status: {task_obj.status.value}\n"
                )

        result_msg = "\n".join(result_lines)

        agent.short_memory.add(
            role="Sub-Agent Execution",
            content=result_msg,
        )

        if agent.verbose:
            logger.info(
                f"Completed {len(task_ids)} sub-agent task(s)"
            )

        return result_msg

    except Exception as e:
        error_msg = f"Error assigning tasks to sub-agents: {str(e)}"
        logger.error(error_msg)
        agent.short_memory.add(
            role="Sub-Agent Execution",
            content=f"Error: {error_msg}",
        )
        return error_msg


def get_task_status_tool(
    agent: Any,
    task_ids: Optional[List[str]] = None,
    **kwargs,
) -> str:
    """Get status of running/completed subagent tasks."""
    try:
        registry = _get_task_registry(agent)
        if not registry.tasks:
            return "No subagent tasks have been submitted yet."

        tasks_to_check = {}
        if task_ids:
            for tid in task_ids:
                try:
                    tasks_to_check[tid] = registry.get_task(tid)
                except KeyError:
                    pass
        else:
            tasks_to_check = registry.tasks

        lines = [
            f"Subagent task status ({len(tasks_to_check)} tasks):"
        ]
        for tid, task in tasks_to_check.items():
            duration = ""
            if task.completed_at:
                duration = (
                    f" ({task.completed_at - task.created_at:.2f}s)"
                )
            lines.append(
                f"  [{task.agent_name}] {tid}: {task.status.value}{duration}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error getting task status: {e}"


def cancel_task_tool(
    agent: Any, task_id: str, **kwargs
) -> str:
    """Cancel a running subagent task."""
    try:
        registry = _get_task_registry(agent)
        if registry.cancel(task_id):
            return f"Task {task_id} cancelled successfully."
        return f"Task {task_id} could not be cancelled (may have already completed)."
    except KeyError:
        return f"Task {task_id} not found."
    except Exception as e:
        return f"Error cancelling task: {e}"
