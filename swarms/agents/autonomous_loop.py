"""
The plan, execute and summarize loop agents run when max_loops is auto.
"""

import json
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger

from swarms.prompts.handoffs_prompt import get_handoffs_prompt
from swarms.structs.autonomous_loop_utils import (
    assign_task_tool,
    cancel_sub_agent_tasks_tool,
    check_sub_agent_status_tool,
    create_file_tool,
    create_sub_agent_tool,
    delete_file_tool,
    get_autonomous_planning_tools,
    get_execution_prompt,
    get_planning_prompt,
    grep_tool,
    list_directory_tool,
    glob_tool,
    read_file_tool,
    respond_to_user_tool,
    run_bash_tool,
    update_file_tool,
)
from swarms.tools.handoffs_tool_schema import get_handoff_tool_schema
from swarms.tools.py_func_to_openai_func_str import (
    convert_multiple_functions_to_openai_function_schema,
)
from swarms.structs.conversation import map_batch_results
from swarms.tools.dynamic_tool_loader import SEARCH_TOOL_NAME
from swarms.utils.formatter import formatter
from swarms.utils.index import exists, format_data_structure


def _format_tool_error(function_name: str, error: Exception) -> str:
    """
    Render a tool failure as text the model can act on.

    Tool errors are fed back into the conversation as the tool's result so the
    model can correct itself on the next turn. Swallowing them means the next
    iteration rebuilds an identical prompt and the model re-emits the identical
    failing call until the iteration budget is gone.
    """
    return (
        f"ERROR: {function_name} failed with "
        f"{type(error).__name__}: {error}. "
        "Review the arguments and either retry with a correction or take a "
        "different approach. Do not repeat the same call unchanged."
    )


# Enough for a typical plan without pulling in the whole catalog.
PREWARM_TOOL_LIMIT = 8

# Pre-warm matches must score at least this fraction of the best match.
PREWARM_MIN_SCORE_RATIO = 0.6

# Never deferred: searching for subtask_done would stall the loop.
ALWAYS_LOADED_TOOLS = frozenset(
    {
        "create_plan",
        "think",
        "subtask_done",
        "complete_task",
        "respond_to_user",
    }
)


class AutonomousAgentLoop:
    """
    Plan, execute and summarize loop used when max_loops is auto.

    Args:
        agent (Any): The agent whose state and configuration the loop uses.
    """

    def __init__(self, agent: Any):
        self.agent = agent
        # Index of this run's first row in short_memory; the request starts there.
        self._window_start = 0
        # Removed before the next append, so runs do not stack copies.
        self._applied_handoff_block: Optional[str] = None

    def _say_user(self, content: str) -> None:
        """
        Add a user turn.

        Args:
            content (str): The message.
        """
        self.agent.short_memory.add(
            role=self.agent.user_name, content=content
        )

    def _messages(self) -> List[Dict[str, Any]]:
        """
        Return this run's conversation as the request body.

        Returns:
            List[Dict[str, Any]]: Chat-completions messages.
        """
        return self.agent.short_memory.to_messages(
            self.agent.agent_name, start=self._window_start
        )

    def _maybe_compress_context(self) -> bool:
        """
        Compress the conversation when it nears the context limit.

        Returns:
            bool: True when it compressed, so the current instruction can be resent.
        """
        compressor = getattr(self.agent, "_context_compressor", None)
        if compressor is None:
            return False
        if compressor.maybe_compress(self.agent) is None:
            return False

        # compact() leaves the summary last; the rows before it are static context.
        self._window_start = (
            len(self.agent.short_memory.conversation_history) - 1
        )
        return True

    def _run_autonomous_loop(
        self,
        task: Optional[Union[str, Any]] = None,
        img: Optional[str] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        *args,
        **kwargs,
    ) -> Any:
        """
        Plan the task, execute each subtask, then summarize.

        Args:
            task (Optional[Union[str, Any]]): The task to complete.
            img (Optional[str]): An image to send with the task.
            streaming_callback (Optional[Callable[[str], None]]): Receives tokens.
            messages (Optional[List[Dict[str, Any]]]): Prior turns to continue from.
            *args: Passed through to the model calls.
            **kwargs: Passed through to the model calls.

        Returns:
            Any: The agent's output, shaped by its output type.
        """
        try:

            # run() has just added the caller's messages; the request starts with them.
            self._window_start = len(
                self.agent.short_memory.conversation_history
            ) - len(messages or [])
            self.agent.autonomous_subtasks = []
            self.agent.current_subtask_index = 0
            self.agent.subtask_status = {}
            self.agent.plan_created = False
            self.agent.think_call_count = 0

            self._say_user(task)

            # Add planning tools to tools_list_dictionary
            planning_tools = get_autonomous_planning_tools()

            # Filter planning tools if selected_tools is not "all"
            if (
                self.agent.selected_tools != "all"
                and self.agent.selected_tools is not None
            ):
                logger.info(
                    f"Filtering autonomous looper tools to: {self.agent.selected_tools}"
                )
                filtered_tools = []
                for tool in planning_tools:
                    tool_name = tool.get("function", {}).get(
                        "name", ""
                    )
                    if tool_name in self.agent.selected_tools:
                        filtered_tools.append(tool)
                planning_tools = filtered_tools
                logger.info(
                    f"Filtered to {len(planning_tools)} tools: {[t.get('function', {}).get('name', '') for t in planning_tools]}"
                )

            # Opt-in; the old thinking_tokens check was always true.
            if not getattr(self.agent, "think_tool", False):
                planning_tools = [
                    t
                    for t in planning_tools
                    if t.get("function", {}).get("name") != "think"
                ]
            elif self.agent.thinking_tokens:
                logger.info(
                    "think_tool=True alongside thinking_tokens="
                    f"{self.agent.thinking_tokens}: the model reasons natively, "
                    "so the think tool adds a round-trip without adding "
                    "information. Consider think_tool=False."
                )

            if self.agent.tools_list_dictionary is None:
                self.agent.tools_list_dictionary = []

            # Only control tools stay; the rest load via tool_search.
            if self.agent.dynamic_tools:
                control = [
                    t
                    for t in planning_tools
                    if t.get("function", {}).get("name")
                    in ALWAYS_LOADED_TOOLS
                ]
                deferred = [
                    t for t in planning_tools if t not in control
                ]
                self.agent.setup_dynamic_tools(always_loaded=control)
                self.agent.defer_tool_schemas(deferred)
                planning_tools = []

            # Get existing tool names to avoid duplicates
            existing_tool_names = set()
            if self.agent.tools_list_dictionary:
                for tool in self.agent.tools_list_dictionary:
                    if isinstance(tool, dict) and "function" in tool:
                        existing_tool_names.add(
                            tool["function"].get("name", "")
                        )

            # Add planning tools (avoid duplicates)
            for tool in planning_tools:
                tool_name = tool.get("function", {}).get("name", "")
                if tool_name not in existing_tool_names:
                    self.agent.tools_list_dictionary.append(tool)
                    existing_tool_names.add(tool_name)

            # Add handoff tool if handoffs are configured (avoid duplicates)
            if exists(self.agent.handoffs):
                handoff_tool_schema = get_handoff_tool_schema()
                for tool in handoff_tool_schema:
                    tool_name = tool.get("function", {}).get(
                        "name", ""
                    )
                    if tool_name not in existing_tool_names:
                        self.agent.tools_list_dictionary.append(tool)
                        existing_tool_names.add(tool_name)

                # Removed first, so a changed roster cannot go stale.
                agent_registry = self.agent._get_agent_registry()
                if agent_registry:
                    handoff_prompt = get_handoffs_prompt(
                        list(agent_registry.values())
                    )
                    handoff_block = "\n\n" + handoff_prompt

                    previous_block = self._applied_handoff_block
                    if (
                        previous_block
                        and previous_block in self.agent.system_prompt
                    ):
                        self.agent.system_prompt = (
                            self.agent.system_prompt.replace(
                                previous_block, "", 1
                            )
                        )

                    # Left alone if present for another reason.
                    if handoff_block not in self.agent.system_prompt:
                        self.agent.system_prompt += handoff_block

                    self._applied_handoff_block = handoff_block

            # Reinitialize LLM with planning tools (and handoff tool if configured)
            if self.agent.llm is not None:
                self.agent.llm = self.agent.llm_handling()

            # Register planning tool handlers
            all_planning_tool_handlers = {
                SEARCH_TOOL_NAME: self.agent._tool_search_tool,
                "create_plan": self._create_plan_tool,
                "think": self._think_tool,
                "subtask_done": self._subtask_done_tool,
                "complete_task": self.agent._complete_task_tool,
                "respond_to_user": lambda **kwargs: respond_to_user_tool(
                    self.agent, **kwargs
                ),
                "create_file": lambda **kwargs: create_file_tool(
                    self.agent, **kwargs
                ),
                "update_file": lambda **kwargs: update_file_tool(
                    self.agent, **kwargs
                ),
                "read_file": lambda **kwargs: read_file_tool(
                    self.agent, **kwargs
                ),
                "list_directory": lambda **kwargs: list_directory_tool(
                    self.agent, **kwargs
                ),
                "delete_file": lambda **kwargs: delete_file_tool(
                    self.agent, **kwargs
                ),
                "run_bash": lambda **kwargs: run_bash_tool(
                    self.agent, **kwargs
                ),
                "grep": lambda **kwargs: grep_tool(
                    self.agent, **kwargs
                ),
                "glob": lambda **kwargs: glob_tool(
                    self.agent, **kwargs
                ),
                "create_sub_agent": lambda **kwargs: create_sub_agent_tool(
                    self.agent, **kwargs
                ),
                "assign_task": lambda **kwargs: assign_task_tool(
                    self.agent, **kwargs
                ),
                "check_sub_agent_status": lambda **kwargs: check_sub_agent_status_tool(
                    self.agent, **kwargs
                ),
                "cancel_sub_agent_tasks": lambda **kwargs: cancel_sub_agent_tasks_tool(
                    self.agent, **kwargs
                ),
            }

            # Filter tool handlers if selected_tools is not "all"
            if (
                self.agent.selected_tools != "all"
                and self.agent.selected_tools is not None
            ):
                planning_tool_handlers = {
                    k: v
                    for k, v in all_planning_tool_handlers.items()
                    if k in self.agent.selected_tools
                }
            else:
                planning_tool_handlers = all_planning_tool_handlers

            # Add handoff tool handler if handoffs are configured
            if exists(self.agent.handoffs):
                planning_tool_handlers["handoff_task"] = (
                    self.agent._handoff_task_tool
                )

            # Phase 1: Planning
            if self.agent.print_on:
                formatter.print_panel(
                    f"Starting planning phase for task:\n\n{task}",
                    title="Autonomous Loop: Planning Phase",
                )

            planning_prompt = get_planning_prompt(task)
            self._say_user(planning_prompt)

            plan_created = False
            planning_attempts = 0
            max_planning_attempts = self.agent.max_planning_attempts

            while (
                not plan_created
                and planning_attempts < max_planning_attempts
            ):
                planning_attempts += 1
                try:
                    response = self.agent.call_llm(
                        task=None,
                        img=img,
                        current_loop=0,
                        streaming_callback=streaming_callback,
                        messages=self._messages(),
                        *args,
                        **kwargs,
                    )

                    response = self.agent.parse_llm_output(response)
                    planning_calls = (
                        self.agent.short_memory.record_assistant(
                            self.agent.agent_name, response
                        )
                    )
                    planning_results: Dict[str, Any] = {}

                    # Check if response contains create_plan or handoff_task tool call
                    if isinstance(response, list):
                        for tool_call in response:
                            if isinstance(tool_call, dict):
                                function_name = tool_call.get(
                                    "function", {}
                                ).get("name")

                                if function_name == "create_plan":
                                    # Execute create_plan tool
                                    arguments = json.loads(
                                        tool_call["function"][
                                            "arguments"
                                        ]
                                    )

                                    # Visualize function call
                                    self.agent._visualize_function_call(
                                        "create_plan", arguments
                                    )

                                    result = planning_tool_handlers[
                                        "create_plan"
                                    ](**arguments)
                                    planning_results[
                                        tool_call.get("id", "")
                                    ] = result

                                elif (
                                    function_name == "handoff_task"
                                    and exists(self.agent.handoffs)
                                ):
                                    # Handle handoff tool call in planning phase
                                    arguments = json.loads(
                                        tool_call["function"][
                                            "arguments"
                                        ]
                                    )
                                    handoffs_list = arguments.get(
                                        "handoffs", []
                                    )

                                    # Visualize handoff tool call
                                    if self.agent.print_on:
                                        self.agent._visualize_handoff_call(
                                            handoffs_list, tool_call
                                        )

                                    result = (
                                        self.agent._handoff_task_tool(
                                            handoffs=handoffs_list
                                        )
                                    )
                                    planning_results[
                                        tool_call.get("id", "")
                                    ] = result

                                # Show plan creation result
                                if self.agent.print_on:
                                    plan_summary = f"Plan created with {len(self.agent.autonomous_subtasks)} subtasks:\n\n"
                                    for i, subtask in enumerate(
                                        self.agent.autonomous_subtasks,
                                        1,
                                    ):
                                        plan_summary += f"{i}. {subtask['step_id']}: {subtask['description']}\n"
                                        plan_summary += f"   Priority: {subtask['priority']}\n"
                                        if subtask.get(
                                            "dependencies"
                                        ):
                                            plan_summary += f"   Dependencies: {', '.join(subtask['dependencies'])}\n"

                                    formatter.print_panel(
                                        plan_summary,
                                        title="Plan Created",
                                    )

                                plan_created = True
                                break

                    # Answer every tool_call before the next request.
                    self.agent.short_memory.flush_tool_results(
                        planning_calls, planning_results
                    )

                    # Also check if plan was created via tool execution
                    if self.agent.plan_created:
                        plan_created = True
                        break

                except Exception as e:
                    if self.agent.verbose:
                        logger.error(
                            f"Error in planning phase (attempt {planning_attempts}): {e}"
                        )
                    if planning_attempts >= max_planning_attempts:
                        raise

            if not plan_created:
                raise Exception(
                    "Failed to create plan after maximum attempts"
                )

            # Already in the catalog when dynamic_tools is on.
            if (
                exists(self.agent.tools)
                and not self.agent.dynamic_tools
            ):
                # Convert user tools to function schema
                user_tools = convert_multiple_functions_to_openai_function_schema(
                    self.agent.tools
                )

                # Get existing tool names to avoid duplicates
                existing_tool_names = set()
                if self.agent.tools_list_dictionary:
                    for tool in self.agent.tools_list_dictionary:
                        if (
                            isinstance(tool, dict)
                            and "function" in tool
                        ):
                            existing_tool_names.add(
                                tool["function"].get("name", "")
                            )

                # Add user tools to tools_list_dictionary (avoid duplicates)
                if self.agent.tools_list_dictionary is None:
                    self.agent.tools_list_dictionary = []

                tools_added = 0
                for tool in user_tools:
                    tool_name = tool.get("function", {}).get(
                        "name", ""
                    )
                    if tool_name not in existing_tool_names:
                        self.agent.tools_list_dictionary.append(tool)
                        existing_tool_names.add(tool_name)
                        tools_added += 1

                # Reinitialize LLM with both planning tools and user tools
                if self.agent.llm is not None:
                    self.agent.llm = self.agent.llm_handling()

                if self.agent.print_on and tools_added > 0:
                    formatter.print_panel(
                        f"Integrated {tools_added} user tools into autonomous loop",
                        title="Tools Integration",
                    )

            # Phase 2: Execution - For each subtask
            if self.agent.print_on:
                formatter.print_panel(
                    f"Starting execution phase with {len(self.agent.autonomous_subtasks)} subtasks",
                    title="Autonomous Loop: Execution Phase",
                )

            max_subtask_iterations = self.agent.max_subtask_iterations
            total_iterations = 0

            while not self._all_subtasks_complete():
                total_iterations += 1
                if total_iterations > max_subtask_iterations:
                    if self.agent.print_on:
                        formatter.print_panel(
                            f"Maximum iterations ({max_subtask_iterations}) reached. Stopping execution.",
                            title="Execution Limit Reached",
                        )
                    if self.agent.verbose:
                        logger.warning(
                            f"Maximum iterations ({max_subtask_iterations}) reached. Stopping execution."
                        )
                    break

                # Get next executable subtask
                current_subtask = self._get_next_executable_subtask()
                if current_subtask is None:
                    # All subtasks are done or blocked
                    if self._all_subtasks_complete():
                        break
                    else:
                        if self.agent.verbose:
                            logger.warning(
                                "No executable subtasks found, but not all are complete"
                            )
                        break

                subtask_id = current_subtask["step_id"]
                subtask_desc = current_subtask["description"]
                subtask_priority = current_subtask.get(
                    "priority", "medium"
                )

                # Show subtask start
                if self.agent.print_on:
                    progress = f"{sum(1 for s in self.agent.autonomous_subtasks if s['status'] in ['completed', 'failed', 'skipped'])}/{len(self.agent.autonomous_subtasks)}"
                    formatter.print_panel(
                        f"Subtask: {subtask_id}\nDescription: {subtask_desc}\nPriority: {subtask_priority}\nProgress: {progress} subtasks completed",
                        title=f"Executing Subtask: {subtask_id}",
                    )

                # Subtask execution loop: thinking -> tool actions -> observation
                subtask_iterations = 0
                max_subtask_loops = self.agent.max_subtask_loops
                subtask_done = False

                # Consecutive across the subtask, not one response.
                self.agent.think_call_count = 0

                # Once only, or the model reads duplicates as a rerun.
                execution_prompt = get_execution_prompt(
                    subtask_id,
                    subtask_desc,
                    self.agent.autonomous_subtasks,
                )
                self._say_user(execution_prompt)

                while (
                    not subtask_done
                    and subtask_iterations < max_subtask_loops
                ):
                    subtask_iterations += 1

                    # Every tool call is answered here, so replacing the transcript orphans nothing
                    if self._maybe_compress_context():
                        # The rebuilt transcript holds only the summary, restore the subtask
                        self._say_user(execution_prompt)

                    # Before the try, so the except can answer calls left open.
                    turn_calls: List[Dict[str, Any]] = []
                    turn_results: Dict[str, Any] = {}
                    try:
                        response = self.agent.call_llm(
                            task=None,
                            img=img,
                            current_loop=subtask_iterations,
                            streaming_callback=streaming_callback,
                            messages=self._messages(),
                            *args,
                            **kwargs,
                        )

                        response = self.agent.parse_llm_output(
                            response
                        )

                        # Answer every call before the next request.
                        turn_calls = (
                            self.agent.short_memory.record_assistant(
                                self.agent.agent_name, response
                            )
                        )

                        # Handle tool calls
                        if isinstance(response, list):
                            regular_tool_calls = []
                            # Set, not returned, so later calls run.
                            task_complete = False

                            for tool_call in response:
                                if isinstance(
                                    tool_call, dict
                                ) and tool_call.get(
                                    "function", {}
                                ).get(
                                    "name"
                                ):
                                    function_name = tool_call[
                                        "function"
                                    ]["name"]
                                    try:
                                        arguments = json.loads(
                                            tool_call["function"][
                                                "arguments"
                                            ]
                                        )
                                    except (
                                        json.JSONDecodeError,
                                        TypeError,
                                    ) as parse_error:
                                        # Report back, do not abort.
                                        turn_results[
                                            tool_call.get("id", "")
                                        ] = _format_tool_error(
                                            function_name,
                                            parse_error,
                                        )
                                        if self.agent.verbose:
                                            logger.warning(
                                                f"Could not parse arguments for {function_name}: {parse_error}"
                                            )
                                        continue

                                    # Handle planning tools and handoff tool
                                    if (
                                        function_name
                                        in planning_tool_handlers
                                    ):
                                        # A raise is not a completion.
                                        tool_failed = False

                                        # Special handling for handoff_task tool
                                        if (
                                            function_name
                                            == "handoff_task"
                                        ):
                                            # Visualize handoff tool call
                                            handoffs_list = (
                                                arguments.get(
                                                    "handoffs", []
                                                )
                                            )
                                            if self.agent.print_on:
                                                self.agent._visualize_handoff_call(
                                                    handoffs_list,
                                                    tool_call,
                                                )

                                            try:
                                                result = self.agent._handoff_task_tool(
                                                    handoffs=handoffs_list
                                                )
                                            except (
                                                Exception
                                            ) as tool_error:
                                                tool_failed = True
                                                result = _format_tool_error(
                                                    function_name,
                                                    tool_error,
                                                )
                                        else:
                                            # Shown post-execution.
                                            if function_name not in (
                                                "subtask_done",
                                                "complete_task",
                                            ):
                                                self.agent._visualize_function_call(
                                                    function_name,
                                                    arguments,
                                                )

                                            try:
                                                result = planning_tool_handlers[
                                                    function_name
                                                ](
                                                    **arguments
                                                )
                                            except (
                                                Exception
                                            ) as tool_error:
                                                tool_failed = True
                                                result = _format_tool_error(
                                                    function_name,
                                                    tool_error,
                                                )

                                        turn_results[
                                            tool_call.get("id", "")
                                        ] = result

                                        # Non-think breaks the streak.
                                        if function_name != "think":
                                            self.agent.think_call_count = (
                                                0
                                            )

                                        if tool_failed:
                                            if self.agent.print_on:
                                                formatter.print_panel(
                                                    result,
                                                    title=f"Tool Error: {function_name}",
                                                )
                                            if self.agent.verbose:
                                                logger.warning(result)

                                        # Visualize result for important tools
                                        if function_name in [
                                            "subtask_done",
                                            "complete_task",
                                        ]:
                                            self.agent._visualize_function_call(
                                                function_name,
                                                arguments,
                                                result,
                                            )

                                        # A failure completes nothing.
                                        if (
                                            function_name
                                            == "subtask_done"
                                            and not tool_failed
                                        ):
                                            if (
                                                arguments.get(
                                                    "task_id"
                                                )
                                                == subtask_id
                                            ):
                                                subtask_done = True
                                                # Show subtask completion
                                                if (
                                                    self.agent.print_on
                                                ):
                                                    status = (
                                                        "completed"
                                                        if arguments.get(
                                                            "success"
                                                        )
                                                        else "failed"
                                                    )
                                                    formatter.print_panel(
                                                        f"Subtask {subtask_id} marked as {status}\n\nSummary: {arguments.get('summary', 'N/A')}",
                                                        title=f"Subtask {status.title()}: {subtask_id}",
                                                    )

                                        # Deferred until calls finish.
                                        if (
                                            function_name
                                            == "complete_task"
                                            and not tool_failed
                                        ):
                                            task_complete = True
                                    else:
                                        # Collect regular tool calls for batch visualization and execution
                                        regular_tool_calls.append(
                                            tool_call
                                        )

                            # MCP resolves elsewhere; split first.
                            if regular_tool_calls:
                                (
                                    mcp_calls,
                                    regular_tool_calls,
                                ) = self._split_mcp_calls(
                                    regular_tool_calls
                                )
                                if mcp_calls:
                                    self._execute_mcp_calls(
                                        mcp_calls,
                                        turn_results,
                                        subtask_iterations,
                                    )

                            tool_output = None
                            if regular_tool_calls and exists(
                                self.agent.tools
                            ):
                                # Visualize all regular tool calls first
                                if self.agent.print_on:
                                    for (
                                        tool_call
                                    ) in regular_tool_calls:
                                        func_name = tool_call.get(
                                            "function", {}
                                        ).get("name", "Unknown")
                                        func_args = {}
                                        try:
                                            func_args = json.loads(
                                                tool_call.get(
                                                    "function", {}
                                                ).get(
                                                    "arguments", "{}"
                                                )
                                            )
                                        except (
                                            json.JSONDecodeError,
                                            AttributeError,
                                        ):
                                            pass
                                        self.agent._visualize_function_call(
                                            func_name, func_args
                                        )

                                # Execute all regular tools together
                                try:
                                    tool_output = self.agent.tool_struct.execute_function_calls_from_api_response(
                                        regular_tool_calls
                                    )
                                    map_batch_results(
                                        regular_tool_calls,
                                        tool_output,
                                        turn_results,
                                        formatter=format_data_structure,
                                    )

                                    # Display tool execution results using formatter
                                    if self.agent.print_on:
                                        tool_names = [
                                            tc.get(
                                                "function", {}
                                            ).get("name", "Unknown")
                                            for tc in regular_tool_calls
                                        ]
                                        tool_display = f"Tools Executed: {', '.join(tool_names)}\n\n"
                                        tool_display += f"Output:\n{format_data_structure(tool_output)}"

                                        formatter.print_panel(
                                            tool_display,
                                            title="Tool Execution Results",
                                        )

                                except Exception as e:
                                    # Fallback to tool_execution_retry if direct execution fails
                                    if self.agent.verbose:
                                        logger.warning(
                                            f"Direct tool execution failed, using retry mechanism: {e}"
                                        )
                                    tool_output = self.agent.tool_execution_retry(
                                        regular_tool_calls,
                                        subtask_iterations,
                                    )
                                    map_batch_results(
                                        regular_tool_calls,
                                        tool_output,
                                        turn_results,
                                        formatter=format_data_structure,
                                    )

                            self.agent.short_memory.flush_tool_results(
                                turn_calls, turn_results
                            )
                            turn_calls = []

                            # After the results, so the summary follows what it summarises.
                            if (
                                self.agent.tool_call_summary is True
                                and tool_output
                            ):
                                self.agent._summarize_tool_output(
                                    tool_output, subtask_iterations
                                )

                            if task_complete:
                                return self.agent._generate_final_summary(
                                    streaming_callback=streaming_callback,
                                    messages=self._messages(),
                                )
                        else:
                            # Handle regular tool execution
                            if exists(self.agent.tools):
                                # Visualize tool calls before execution
                                if (
                                    isinstance(response, list)
                                    and self.agent.print_on
                                ):
                                    for tool_call in response:
                                        if isinstance(
                                            tool_call, dict
                                        ):
                                            func_name = tool_call.get(
                                                "function", {}
                                            ).get("name", "Unknown")
                                            func_args = {}
                                            try:
                                                func_args = json.loads(
                                                    tool_call.get(
                                                        "function", {}
                                                    ).get(
                                                        "arguments",
                                                        "{}",
                                                    )
                                                )
                                            except (
                                                json.JSONDecodeError,
                                                AttributeError,
                                            ):
                                                pass

                                            # Only visualize if it's not a planning tool
                                            if (
                                                func_name
                                                not in planning_tool_handlers
                                            ):
                                                self.agent._visualize_function_call(
                                                    func_name,
                                                    func_args,
                                                )

                                # Execute tools and capture output for display
                                tool_output = None
                                try:
                                    tool_output = self.agent.tool_struct.execute_function_calls_from_api_response(
                                        response
                                    )

                                    # Display tool execution results using formatter
                                    if self.agent.print_on:
                                        tool_display = f"Tool Output:\n{format_data_structure(tool_output)}"
                                        formatter.print_panel(
                                            tool_display,
                                            title="Tool Execution Results",
                                        )

                                except Exception as e:
                                    # Fallback to tool_execution_retry if direct execution fails
                                    if self.agent.verbose:
                                        logger.warning(
                                            f"Direct tool execution failed, using retry mechanism: {e}"
                                        )
                                    tool_output = self.agent.tool_execution_retry(
                                        response, subtask_iterations
                                    )

                                # No recorded call to answer, so it is kept for the record only.
                                if tool_output:
                                    self.agent.short_memory.add(
                                        role="Tool Executor",
                                        content=format_data_structure(
                                            tool_output
                                        ),
                                        internal=True,
                                    )
                                    if (
                                        self.agent.tool_call_summary
                                        is True
                                    ):
                                        self.agent._summarize_tool_output(
                                            tool_output,
                                            subtask_iterations,
                                        )

                            self.agent.short_memory.flush_tool_results(
                                turn_calls, turn_results
                            )
                            turn_calls = []

                        # Check if subtask status changed
                        if (
                            subtask_id in self.agent.subtask_status
                            and self.agent.subtask_status[subtask_id]
                            in ["completed", "failed"]
                        ):
                            subtask_done = True

                        # Prevent infinite thinking loops
                        if (
                            self.agent.think_call_count
                            >= self.agent.max_consecutive_thinks
                        ):
                            if self.agent.print_on:
                                formatter.print_panel(
                                    f"Too many consecutive think calls ({self.agent.think_call_count}). Forcing action.",
                                    title="Loop Prevention",
                                )
                            if self.agent.verbose:
                                logger.warning(
                                    f"Too many consecutive think calls ({self.agent.think_call_count}). Forcing action."
                                )
                            # Into the transcript, or it is unseen.
                            nudge = (
                                "You have called `think` "
                                f"{self.agent.think_call_count} times in a row "
                                "without acting. Stop analysing. Take concrete "
                                "action now using the available tools, and call "
                                "subtask_done when the work is finished."
                            )
                            self._say_user(nudge)

                            # Give the nudge a chance before refiring.
                            self.agent.think_call_count = 0

                    except Exception as e:
                        if self.agent.verbose:
                            logger.error(
                                f"Error in subtask execution loop: {e}"
                            )
                        # Without this the next prompt is identical.
                        error = (
                            f"ERROR: the previous step failed with "
                            f"{type(e).__name__}: {e}. Adjust your "
                            "approach before retrying."
                        )
                        if turn_calls:
                            for call in turn_calls:
                                turn_results.setdefault(
                                    call["id"], error
                                )
                            self.agent.short_memory.flush_tool_results(
                                turn_calls, turn_results
                            )
                        else:
                            self.agent.short_memory.add(
                                role="Tool Executor", content=error
                            )

                if not subtask_done:
                    # Failed, not pending; pending would re-run it.
                    reason = (
                        f"Exhausted its {max_subtask_loops}-iteration budget "
                        "without completing."
                    )
                    self.agent.subtask_status[subtask_id] = "failed"
                    for subtask in self.agent.autonomous_subtasks:
                        if subtask["step_id"] == subtask_id:
                            subtask["status"] = "failed"
                            subtask.setdefault("summary", reason)
                            break

                    if self.agent.print_on:
                        formatter.print_panel(
                            f"Subtask {subtask_id} not completed after "
                            f"{max_subtask_loops} iterations - marking failed.",
                            title="Subtask Timeout",
                        )
                    if self.agent.verbose:
                        logger.warning(
                            f"Subtask {subtask_id} not completed after "
                            f"{max_subtask_loops} iterations - marking failed."
                        )

            # Phase 3: Final Summary
            if self.agent.print_on:
                formatter.print_panel(
                    "All subtasks completed. Generating final summary...",
                    title="Autonomous Loop: Summary Phase",
                )

            return self.agent._generate_final_summary(
                streaming_callback=streaming_callback,
                messages=self._messages(),
            )

        except Exception as error:
            self.agent._handle_run_error(error)

    def _create_plan_tool(
        self, task_description: str, steps: List[Dict], **kwargs
    ) -> str:
        """
        Store a plan of subtasks for the run.

        Args:
            task_description (str): The overall task.
            steps (List[Dict]): Subtasks with step_id, description, priority
                and dependencies.
            **kwargs: Ignored.

        Returns:
            str: A confirmation the model reads.
        """
        if self.agent.verbose:
            logger.info(f"Creating plan for task: {task_description}")

        existing = {
            subtask["step_id"]: subtask
            for subtask in self.agent.autonomous_subtasks
        }
        is_revision = bool(existing)

        incoming: Dict[str, Dict[str, Any]] = {}
        incoming_order: List[str] = []
        known_step_ids = {step.get("step_id", "") for step in steps}
        # Finished work stays a valid dependency target.
        known_step_ids |= set(existing)

        for step in steps:
            step_id = step.get("step_id", "")

            # Model-generated ids; drop bad ones, do not deadlock.
            declared = step.get("dependencies", []) or []
            dependencies = [
                dep
                for dep in declared
                if dep in known_step_ids and dep != step_id
            ]
            dangling = [
                dep for dep in declared if dep not in dependencies
            ]
            if dangling:
                logger.warning(
                    f"Subtask {step_id!r} declares unknown or self-referential "
                    f"dependencies {dangling} - dropping them. Known step ids: "
                    f"{sorted(known_step_ids)}"
                )

            incoming[step_id] = {
                "step_id": step_id,
                "description": step.get("description", ""),
                "priority": step.get("priority", "medium"),
                "dependencies": dependencies,
                "status": "pending",
            }
            incoming_order.append(step_id)

        # Keep existing order, append new work at the end.
        merged: List[Dict[str, Any]] = []
        added, updated, removed, retained = [], [], [], []

        for subtask in self.agent.autonomous_subtasks:
            step_id = subtask["step_id"]
            terminal = subtask["status"] in (
                "completed",
                "failed",
                "skipped",
            )
            if step_id in incoming:
                if terminal:
                    # Finished work is not re-opened by a revision.
                    merged.append(subtask)
                    retained.append(step_id)
                else:
                    merged.append(incoming[step_id])
                    updated.append(step_id)
            elif terminal:
                # Not restated, but it happened - keep it as history.
                merged.append(subtask)
                retained.append(step_id)
            else:
                removed.append(step_id)
                self.agent.subtask_status.pop(step_id, None)

        for step_id in incoming_order:
            if step_id not in existing:
                merged.append(incoming[step_id])
                added.append(step_id)

        self.agent.autonomous_subtasks = merged
        for subtask in merged:
            self.agent.subtask_status.setdefault(
                subtask["step_id"], subtask["status"]
            )
            if subtask["status"] == "pending":
                self.agent.subtask_status[subtask["step_id"]] = (
                    "pending"
                )

        self.agent.plan_created = True
        if not is_revision:
            self.agent.current_subtask_index = 0

        if not is_revision:
            if self.agent.verbose:
                logger.info(
                    f"Plan created with {len(merged)} steps: "
                    f"{[s['step_id'] for s in merged]}"
                )
            message = f"Plan created successfully with {len(merged)} subtasks"
            prewarmed = self._prewarm_tools_from_plan(
                task_description, steps
            )
            if prewarmed:
                message += (
                    f". Pre-loaded the tools this plan implies: "
                    f"{', '.join(prewarmed)}. They are callable from your "
                    "next turn - do not search for them again."
                )
            return message

        # Reports the change, not the whole plan.
        diff_parts = []
        if added:
            diff_parts.append(f"added {added}")
        if updated:
            diff_parts.append(f"updated {updated}")
        if removed:
            diff_parts.append(f"removed {removed}")
        if retained:
            diff_parts.append(f"kept finished {retained}")
        summary = "; ".join(diff_parts) or "no changes"

        if self.agent.print_on:
            formatter.print_panel(summary, title="Plan Revised")
        if self.agent.verbose:
            logger.info(f"Plan revised: {summary}")

        message = (
            f"Plan updated ({summary}). The plan now has "
            f"{len(merged)} subtasks."
        )
        prewarmed = self._prewarm_tools_from_plan(
            task_description, steps
        )
        if prewarmed:
            message += (
                f" Pre-loaded for the new steps: "
                f"{', '.join(prewarmed)}."
            )
        return message

    def _mcp_tool_names(self) -> set:
        """Names of the tools the configured MCP servers expose."""
        agent = self.agent
        if not getattr(agent, "mcp_enabled", False):
            return set()

        # Use the loader's cache to avoid a call per turn.
        schemas = getattr(agent, "_mcp_schemas_cache", None)
        if schemas is None:
            try:
                schemas = agent.add_mcp_tools_to_memory()
            except Exception as error:
                logger.error(f"Could not list MCP tools: {error}")
                return set()

        return {
            schema.get("function", {}).get("name")
            for schema in (schemas or [])
            if isinstance(schema, dict)
        }

    def _split_mcp_calls(self, tool_calls: List[Dict[str, Any]]):
        """Partition tool calls into (mcp_calls, everything_else)."""
        mcp_names = self._mcp_tool_names()
        if not mcp_names:
            return [], tool_calls

        mcp_calls, others = [], []
        for call in tool_calls:
            name = (
                call.get("function", {}).get("name")
                if isinstance(call, dict)
                else None
            )
            (mcp_calls if name in mcp_names else others).append(call)
        return mcp_calls, others

    def _execute_mcp_calls(
        self,
        mcp_calls: List[Dict[str, Any]],
        results: Dict[str, Any],
        current_loop: int,
    ) -> None:
        """
        Run MCP tool calls and record each outcome as its result.

        Args:
            mcp_calls (List[Dict[str, Any]]): The MCP calls to run.
            results (Dict[str, Any]): Filled in place, keyed by call id.
            current_loop (int): The current loop, for display.
        """
        for call in mcp_calls:
            name = call.get("function", {}).get("name", "unknown")
            if self.agent.print_on:
                self.agent._visualize_function_call(name, {})

            try:
                output = self.agent.mcp_tool_handling(
                    response=[call], current_loop=current_loop
                )
                outcome = format_data_structure(output)
            except Exception as error:
                outcome = _format_tool_error(name, error)
                if self.agent.verbose:
                    logger.error(outcome)

            results[call.get("id", "")] = outcome

    def _prewarm_tools_from_plan(
        self, task_description: str, steps: List[Dict]
    ) -> List[str]:
        """
        Load the tools a plan implies before any subtask starts.

        Args:
            task_description (str): The overall task.
            steps (List[Dict]): The plan's subtasks.

        Returns:
            List[str]: The names of the tools that were loaded.
        """
        agent = self.agent
        if not getattr(agent, "dynamic_tools", False):
            return []
        if getattr(agent, "tool_loader", None) is None:
            return []

        query = " ".join(
            [task_description or ""]
            + [str(step.get("description", "")) for step in steps]
        )
        before = set(agent.tool_loader.loaded_names)
        agent._tool_search_tool(
            query=query,
            max_results=PREWARM_TOOL_LIMIT,
            # Speculative, so relevance must beat an explicit search.
            min_score_ratio=PREWARM_MIN_SCORE_RATIO,
        )
        return [
            name
            for name in agent.tool_loader.loaded_names
            if name not in before
        ]

    def _think_tool(
        self,
        current_state: str,
        analysis: str,
        next_actions: List[str],
        confidence: float,
        **kwargs,
    ) -> str:
        """
        Analyze current situation and plan next actions.

        This tool allows the agent to pause and think about the current state of
        task execution, analyze the situation, and plan the next steps. It's used
        in the autonomous loop to enable reflective reasoning before taking action.

        **Thinking Process:**
        1. Agent analyzes the current state of execution
        2. Provides reasoning about the situation
        3. Lists potential next actions
        4. Assigns a confidence level to the analysis

        **Loop Prevention:**
        The method tracks consecutive think calls using self.agent.think_call_count.
        If too many consecutive think calls occur (exceeds max_consecutive_thinks),
        the autonomous loop will force action to prevent infinite thinking loops.

        **Memory Integration:**
        The thinking result is added to conversation memory with format:
        "[THINKING] {analysis}\nNext actions: {actions}\nConfidence: {confidence}"

        Args:
            current_state (str): Description of the current state of task execution.
                This should include what has been accomplished and what remains.
            analysis (str): The agent's analysis of the current situation. Should include
                observations, insights, and reasoning about the current state.
            next_actions (List[str]): List of potential next actions to take. Each action
                should be a clear, actionable step the agent can take.
            confidence (float): Confidence level in the analysis, ranging from 0.0 to 1.0.
                Higher values indicate greater confidence in the analysis and planned actions.
            **kwargs: Additional arguments (currently unused, reserved for future use).

        Returns:
            str: Formatted analysis result string containing:
                - Analysis confirmation
                - Confidence level
                - List of next actions

        Note:
            - This method increments self.agent.think_call_count to track consecutive calls
            - Thinking results are automatically added to conversation memory
            - If verbose=True, thinking details are logged
            - Excessive thinking is prevented by max_consecutive_thinks limit

        Examples:
            >>> result = agent._think_tool(
            ...     current_state="Completed step 1, working on step 2",
            ...     analysis="Step 2 requires additional data from step 1",
            ...     next_actions=["Retrieve data from step 1", "Process the data"],
            ...     confidence=0.85
            ... )
            >>> # Returns formatted analysis with confidence and actions
        """
        # Increment think call count
        self.agent.think_call_count += 1

        if self.agent.verbose:
            logger.info(f"Thinking: {analysis}")
            logger.info(f"Next actions: {next_actions}")
            logger.info(f"Confidence: {confidence}")

        result = f"Analysis complete. Confidence: {confidence}. Next actions: {', '.join(next_actions)}"

        # Add to memory
        self.agent.short_memory.add(
            role=self.agent.agent_name,
            content=f"[THINKING] {analysis}\nNext actions: {', '.join(next_actions)}\nConfidence: {confidence}",
            internal=True,
        )

        return result

    def _subtask_done_tool(
        self, task_id: str, summary: str, success: bool, **kwargs
    ) -> str:
        """
        Mark a subtask as completed and move to the next task in the plan.

        This tool is used in the autonomous loop to signal that a subtask has been
        completed (either successfully or with failure). It updates the subtask
        status, stores a summary, and allows the loop to proceed to the next subtask.

        **Status Updates:**
        - Updates self.agent.subtask_status[task_id] to "completed" or "failed"
        - Updates the corresponding subtask in self.agent.autonomous_subtasks
        - Stores the summary in the subtask dictionary
        - Resets think_call_count to allow fresh thinking for next subtask

        **Progress Tracking:**
        - Increments current_subtask_index to move to next subtask
        - The autonomous loop uses this to determine when all subtasks are done

        **Memory Integration:**
        The completion is added to conversation memory with format:
        "[SUBTASK DONE] {task_id}: {summary} (Success: {success})"

        Args:
            task_id (str): The unique identifier (step_id) of the subtask being completed.
                Must match a step_id from the plan created by _create_plan_tool.
            summary (str): A summary of what was accomplished in this subtask. Should
                include key results, findings, or outcomes.
            success (bool): Whether the subtask was completed successfully.
                - True: Subtask completed as intended
                - False: Subtask failed but execution continues
            **kwargs: Additional arguments (currently unused, reserved for future use).

        Returns:
            str: Confirmation message indicating the subtask status. Format:
                "Subtask {task_id} marked as {completed/failed}"

        Note:
            - This method is called automatically by the autonomous loop when a subtask finishes
            - The task_id must exist in autonomous_subtasks
            - Failed subtasks don't block execution but are tracked for final summary
            - Think call count is reset to prevent carryover thinking loops
            - If verbose=True, subtask completion is logged

        Examples:
            >>> result = agent._subtask_done_tool(
            ...     task_id="step1",
            ...     summary="Created project structure with 5 directories",
            ...     success=True
            ... )
            >>> # Returns: "Subtask step1 marked as completed"
            >>> # Updates status and allows loop to proceed to next subtask
        """
        if self.agent.verbose:
            logger.info(f"Completing subtask {task_id}: {summary}")

        # Update subtask status
        if task_id in self.agent.subtask_status:
            self.agent.subtask_status[task_id] = (
                "completed" if success else "failed"
            )

        # Update subtask in list
        for subtask in self.agent.autonomous_subtasks:
            if subtask["step_id"] == task_id:
                subtask["status"] = (
                    "completed" if success else "failed"
                )
                subtask["summary"] = summary
                break

        # Reset think call count when subtask is done
        self.agent.think_call_count = 0

        # Move to next subtask
        self.agent.current_subtask_index += 1

        if self.agent.verbose:
            logger.info(
                f"Subtask {task_id} marked as {'completed' if success else 'failed'}. Moving to next subtask."
            )

        # Add to memory
        self.agent.short_memory.add(
            role=self.agent.agent_name,
            content=f"[SUBTASK DONE] {task_id}: {summary} (Success: {success})",
            internal=True,
        )

        return f"Subtask {task_id} marked as {'completed' if success else 'failed'}"

    def _get_next_executable_subtask(
        self,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the next executable subtask based on dependencies and status.

        Returns:
            Dictionary of the next subtask or None if all are done
        """
        if not self.agent.autonomous_subtasks:
            return None

        # Find subtasks that are pending and have all dependencies completed
        for subtask in self.agent.autonomous_subtasks:
            if subtask["status"] != "pending":
                continue

            dependencies = subtask.get("dependencies", [])
            if not dependencies:
                return subtask

            statuses = [
                self.agent.subtask_status.get(dep)
                for dep in dependencies
            ]

            # Only completed unblocks; failed and unknown do not.
            if all(status == "completed" for status in statuses):
                return subtask

            # Unreachable: skip so the run can terminate.
            blockers = [
                dep
                for dep, status in zip(dependencies, statuses)
                if status in ("failed", "skipped")
            ]
            if blockers:
                self._skip_subtask(subtask, blockers)

        return None

    def _skip_subtask(
        self, subtask: Dict[str, Any], blockers: List[str]
    ) -> None:
        """
        Mark a subtask as skipped because a dependency it needs cannot complete.

        Args:
            subtask: The subtask being skipped.
            blockers: The dependency step_ids that failed or were skipped.
        """
        step_id = subtask["step_id"]
        reason = (
            f"Skipped: depends on {', '.join(blockers)}, which did not "
            "complete successfully."
        )

        subtask["status"] = "skipped"
        subtask["summary"] = reason
        self.agent.subtask_status[step_id] = "skipped"

        if self.agent.print_on:
            formatter.print_panel(
                reason, title=f"Subtask Skipped: {step_id}"
            )
        if self.agent.verbose:
            logger.warning(f"Subtask {step_id} skipped. {reason}")

    def _all_subtasks_complete(self) -> bool:
        """
        Check if all subtasks are completed.

        Returns:
            bool: True if all subtasks are completed or failed
        """
        if not self.agent.autonomous_subtasks:
            return False

        return all(
            subtask["status"] in ["completed", "failed", "skipped"]
            for subtask in self.agent.autonomous_subtasks
        )
