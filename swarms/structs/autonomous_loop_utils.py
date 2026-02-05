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
"""

import asyncio
import os
import subprocess
from typing import Any, Dict, List
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
                "description": "Assign a task to one or more sub-agents. Tasks are executed asynchronously and results are returned to the main agent.",
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
                            "description": "Whether to wait for all tasks to complete before returning (default: true)",
                        },
                    },
                    "required": ["assignments"],
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

        created_agents = []

        for agent_spec in agents:
            agent_name = agent_spec.get("agent_name")
            agent_description = agent_spec.get("agent_description")
            system_prompt = agent_spec.get("system_prompt")

            if not agent_name or not agent_description:
                return "Error: Each agent must have agent_name and agent_description"

            # Generate unique ID for the sub-agent
            import uuid

            agent_id = f"sub-agent-{uuid.uuid4().hex[:8]}"

            # Import Agent class to create sub-agent
            from swarms.structs.agent import Agent

            # Create sub-agent with the same LLM as parent
            sub_agent = Agent(
                id=agent_id,
                agent_name=agent_name,
                agent_description=agent_description,
                system_prompt=system_prompt,  # Use custom system prompt if provided
                model_name=agent.model_name,
                max_loops=1,
                print_on=False,  # Reduce noise from sub-agents
            )

            # Cache the sub-agent
            agent.sub_agents[agent_id] = {
                "agent": sub_agent,
                "name": agent_name,
                "description": agent_description,
                "system_prompt": system_prompt,
                "created_at": str(
                    __import__("datetime").datetime.now()
                ),
            }

            created_agents.append(f"{agent_name} (ID: {agent_id})")

            if agent.verbose:
                logger.info(
                    f"Created sub-agent: {agent_name} with ID {agent_id}"
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
    **kwargs,
) -> str:
    """
    Assign tasks to one or more sub-agents. Tasks are executed asynchronously.

    Args:
        agent: The main agent instance
        assignments: List of task assignments with agent_id and task
        wait_for_completion: Whether to wait for all tasks to complete
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
                return f"Error: Sub-agent with ID '{agent_id}' not found. Available agents: {list(agent.sub_agents.keys())}"

        # Prepare tasks for async execution
        async def run_agent_task(
            sub_agent_data: Dict, task: str, task_id: str
        ):
            """Run a single agent task asynchronously."""
            try:
                sub_agent = sub_agent_data["agent"]
                result = await asyncio.to_thread(sub_agent.run, task)
                return {
                    "agent_id": sub_agent.id,
                    "agent_name": sub_agent_data["name"],
                    "task_id": task_id,
                    "status": "success",
                    "result": result,
                }
            except Exception as e:
                return {
                    "agent_id": sub_agent.id,
                    "agent_name": sub_agent_data["name"],
                    "task_id": task_id,
                    "status": "error",
                    "error": str(e),
                }

        async def run_all_tasks():
            """Run all tasks concurrently."""
            tasks = []
            for idx, assignment in enumerate(assignments):
                agent_id = assignment.get("agent_id")
                task = assignment.get("task")
                task_id = assignment.get("task_id", f"task-{idx + 1}")

                sub_agent_data = agent.sub_agents[agent_id]
                tasks.append(
                    run_agent_task(sub_agent_data, task, task_id)
                )

            return await asyncio.gather(*tasks)

        # Execute tasks
        if wait_for_completion:
            # Run in event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            results = loop.run_until_complete(run_all_tasks())

            # Format results
            result_lines = [
                f"Completed {len(results)} task assignment(s):\n"
            ]
            for result in results:
                if result["status"] == "success":
                    result_lines.append(
                        f"\n[{result['agent_name']}] Task {result['task_id']}:"
                    )
                    result_lines.append(
                        f"Result: {result['result']}\n"
                    )
                else:
                    result_lines.append(
                        f"\n[{result['agent_name']}] Task {result['task_id']} FAILED:"
                    )
                    result_lines.append(f"Error: {result['error']}\n")

            result_msg = "\n".join(result_lines)

            # Add to memory
            agent.short_memory.add(
                role="Sub-Agent Execution",
                content=result_msg,
            )

            if agent.verbose:
                logger.info(
                    f"Completed {len(results)} sub-agent task(s)"
                )

            return result_msg
        else:
            # Fire and forget
            asyncio.create_task(run_all_tasks())
            return f"Dispatched {len(assignments)} task(s) to sub-agents (async mode)"

    except Exception as e:
        error_msg = f"Error assigning tasks to sub-agents: {str(e)}"
        logger.error(error_msg)
        agent.short_memory.add(
            role="Sub-Agent Execution",
            content=f"Error: {error_msg}",
        )
        return error_msg
