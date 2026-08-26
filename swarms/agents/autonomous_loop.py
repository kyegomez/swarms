"""
Autonomous execution loop for :class:`~swarms.structs.agent.Agent`.

When an agent is constructed with ``max_loops="auto"`` it does not run a fixed
number of iterations. Instead it plans, executes its plan subtask by subtask,
and decides for itself when the work is finished. That behaviour lives here
rather than on ``Agent`` so the agent class stays readable: this is roughly a
fifth of what ``agent.py`` used to be, and none of it is reachable unless
``max_loops == "auto"``.

The loop owns three phases:

1. **Plan** -- ask the model to produce a subtask list via the ``create_plan`` tool.
2. **Execute** -- walk the subtasks, dispatching tool calls until each is marked
   done via ``subtask_done``.
3. **Summarize** -- hand back a final answer (delegated to
   ``Agent._generate_final_summary``).

``AutonomousAgentLoop`` holds a back-reference to its owning agent and reads and
writes agent state through it, mirroring the existing
:class:`~swarms.agents.llm_manager.LLMManager` arrangement.
"""

import json
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger

from swarms.prompts.handoffs_prompt import get_handoffs_prompt
from swarms.structs.autonomous_loop_utils import (
    MAX_PLANNING_ATTEMPTS,
    MAX_SUBTASK_ITERATIONS,
    MAX_SUBTASK_LOOPS,
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
    read_file_tool,
    respond_to_user_tool,
    run_bash_tool,
    update_file_tool,
)
from swarms.tools.handoffs_tool_schema import get_handoff_tool_schema
from swarms.tools.py_func_to_openai_func_str import (
    convert_multiple_functions_to_openai_function_schema,
)
from swarms.structs.transcript import Transcript
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


# Tools the loop itself depends on. These are never deferred - an agent that
# has to search for its own `subtask_done` cannot finish a subtask.
# How many tools a plan may pre-load in one go. Enough to cover a typical plan
# without pulling in the whole catalog and undoing the saving.
PREWARM_TOOL_LIMIT = 8

# Pre-warm matches must score at least this fraction of the best match.
PREWARM_MIN_SCORE_RATIO = 0.6

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
    Plan-execute-summarize loop used when ``max_loops="auto"``.

    Args:
        agent: The owning :class:`~swarms.structs.agent.Agent`. All agent
            configuration and state is read and written through this
            reference.

    Example:
        >>> agent = Agent(agent_name="Researcher", max_loops="auto")
        >>> agent.run("Compare the top 3 vector databases")  # routes here
    """

    def __init__(self, agent: Any):
        self.agent = agent
        # The real conversation body sent to the model: user turns, assistant
        # turns carrying `tool_calls`, and `{"role": "tool", ...}` results.
        # `short_memory` is kept in sync alongside it because persistence,
        # output formatting and the final summary all read from there.
        self._transcript = Transcript()

    def _say_user(self, content: str, mirror: bool = True) -> None:
        """Add a user turn to the transcript (and to short_memory)."""
        self._transcript.append_user(content)
        if mirror:
            self.agent.short_memory.add(
                role=self.agent.user_name, content=content
            )

    def _record_assistant(self, parsed: Any) -> List[Dict[str, Any]]:
        """Add the model's turn; return the tool calls it made."""
        return self._transcript.record_assistant(parsed)

    def _flush_tool_results(
        self, calls: List[Dict[str, Any]], results: Dict[str, Any]
    ) -> None:
        """Answer every tool call in the preceding assistant turn."""
        self._transcript.flush_tool_results(calls, results)

    def _map_batch_results(
        self,
        tool_calls: List[Dict[str, Any]],
        output: Any,
        results: Dict[str, Any],
    ) -> None:
        """Attribute a batched tool execution back to individual call ids."""
        self._transcript.map_batch_results(
            tool_calls,
            output,
            results,
            formatter=format_data_structure,
        )

    def _run_autonomous_loop(
        self,
        task: Optional[Union[str, Any]] = None,
        img: Optional[str] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute the autonomous loop structure: plan -> execute subtasks -> summary.

        This method implements the optimized autonomous looping when max_loops="auto"
        and interactive=False. It follows a three-phase structure:

        **Phase 1: Planning**
        - Creates a detailed plan using the `create_plan` tool
        - Breaks down the task into subtasks with dependencies, priorities, and step IDs
        - Supports handoff delegation during planning if handoffs are configured
        - Maximum planning attempts are controlled by MAX_PLANNING_ATTEMPTS

        **Phase 2: Execution**
        - Executes each subtask in dependency order
        - For each subtask, runs a thinking -> tool actions -> observation loop
        - Supports both planning tools (think, subtask_done, complete_task) and user-defined tools
        - Prevents infinite thinking loops with max_consecutive_thinks limit
        - Each subtask has a maximum iteration limit (MAX_SUBTASK_LOOPS)
        - Overall execution has a maximum iteration limit (MAX_SUBTASK_ITERATIONS)

        **Phase 3: Summary**
        - Generates a comprehensive final summary when all subtasks are complete
        - Uses the `complete_task` tool or generates summary manually
        - Returns formatted output based on output_type configuration

        The method automatically integrates:
        - Planning tools (create_plan, think, subtask_done, complete_task, file operations)
        - Handoff tools (if handoffs are configured)
        - User-defined tools (added after planning phase)
        - MCP tools (if configured)

        Args:
            task (Optional[Union[str, Any]]): The task or prompt for the agent to process.
                This is the main objective that will be broken down into subtasks.
            img (Optional[str]): Optional image path or data to be processed during execution.
            streaming_callback (Optional[Callable[[str], None]]): Optional callback function
                to receive streaming tokens in real-time. Useful for dashboard integration.
            *args: Additional positional arguments passed to LLM calls.
            **kwargs: Additional keyword arguments passed to LLM calls.

        Returns:
            Any: The agent's output with comprehensive summary. Format depends on output_type:
                - "final": Returns comprehensive task completion summary
                - Other types: Returns formatted conversation history based on output_type

        Raises:
            Exception: If planning phase fails after maximum attempts.
            Exception: If execution exceeds maximum iteration limits.

        Note:
            - This method is automatically called when max_loops="auto" and interactive=False
            - The method resets autonomous loop state at the start of each execution
            - Tool execution results are automatically added to conversation memory
            - Progress visualization is shown if print_on=True

        Examples:
            >>> agent = Agent(max_loops="auto", interactive=False)
            >>> result = agent.run("Build a web application with authentication")
            >>> # The agent will:
            >>> # 1. Create a plan with subtasks
            >>> # 2. Execute each subtask with tool calls
            >>> # 3. Generate a comprehensive summary
        """
        try:

            # Reset autonomous loop state. The transcript is cleared before
            # the task is seeded, or the opening turn would be discarded.
            self._transcript = Transcript()
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

            # The `think` tool is opt-in via Agent(think_tool=True). It costs a
            # full round-trip to produce reasoning the model could emit inline
            # alongside its actions, so it is off unless asked for.
            #
            # This replaces an earlier `thinking_tokens is not None` check.
            # `thinking_tokens` defaults to 1024 rather than None, so that test
            # was always true and silently stripped `think` for every agent -
            # the intent was to drop it only when extended thinking was on.
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

            # With dynamic_tools, only the control tools are always present.
            # The rest of the loop's tools - file, shell, grep, sub-agents -
            # go into the catalog and are loaded on demand via tool_search.
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

                # Add handoff prompt to system prompt
                agent_registry = self.agent._get_agent_registry()
                if agent_registry:
                    handoff_prompt = get_handoffs_prompt(
                        list(agent_registry.values())
                    )
                    self.agent.system_prompt += (
                        "\n\n" + handoff_prompt
                    )

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
            max_planning_attempts = MAX_PLANNING_ATTEMPTS

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
                        messages=self._transcript.messages,
                        *args,
                        **kwargs,
                    )

                    response = self.agent.parse_llm_output(response)
                    self.agent.short_memory.add(
                        role=self.agent.agent_name, content=response
                    )
                    planning_calls = self._record_assistant(response)
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

                                    # Add result to memory
                                    self.agent.short_memory.add(
                                        role="Tool Executor",
                                        content=f"create_plan result: {result}",
                                    )
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

                                    # Add result to memory
                                    self.agent.short_memory.add(
                                        role="Tool Executor",
                                        content=f"handoff_task result: {result}",
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

                    # Every tool_call in the assistant turn must be answered
                    # before the next request, whether or not the plan landed.
                    self._flush_tool_results(
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

            # Integrate user tools after planning phase. With dynamic_tools
            # they are already in the catalog, reachable through tool_search.
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

            max_subtask_iterations = MAX_SUBTASK_ITERATIONS
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
                max_subtask_loops = MAX_SUBTASK_LOOPS
                subtask_done = False

                # Counts CONSECUTIVE think calls across this subtask's
                # iterations. It is reset here, once per subtask, and again by
                # any non-think tool call below - not once per iteration, which
                # would only ever catch repeats inside a single response.
                self.agent.think_call_count = 0

                # Add the execution prompt ONCE before the inner loop so the model
                # doesn't see duplicate copies of it on subsequent iterations and
                # mistakenly conclude "this task has been run before."
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

                    try:
                        response = self.agent.call_llm(
                            task=None,
                            img=img,
                            current_loop=subtask_iterations,
                            streaming_callback=streaming_callback,
                            messages=self._transcript.messages,
                            *args,
                            **kwargs,
                        )

                        response = self.agent.parse_llm_output(
                            response
                        )
                        self.agent.short_memory.add(
                            role=self.agent.agent_name,
                            content=response,
                        )

                        # Record the model's turn, then answer every tool call
                        # it made before the next request goes out.
                        turn_calls = self._record_assistant(response)
                        turn_results: Dict[str, Any] = {}

                        # Handle tool calls
                        if isinstance(response, list):
                            regular_tool_calls = []
                            # complete_task sets this instead of returning
                            # mid-loop, so tool calls batched after it are not
                            # dropped.
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
                                        # Report the malformed payload back to
                                        # the model instead of aborting the
                                        # whole iteration on one bad call.
                                        self.agent.short_memory.add(
                                            role="Tool Executor",
                                            content=_format_tool_error(
                                                function_name,
                                                parse_error,
                                            ),
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
                                        # Set when the handler raises, so a
                                        # failed call is not mistaken for a
                                        # completed subtask or a finished task.
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
                                            # Only pre-visualize tools that won't be shown again
                                            # with their result (subtask_done / complete_task are
                                            # visualized post-execution so skip the pre call).
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

                                        # Add result to memory
                                        self.agent.short_memory.add(
                                            role="Tool Executor",
                                            content=f"{function_name} result: {result}",
                                        )
                                        turn_results[
                                            tool_call.get("id", "")
                                        ] = result

                                        # Any tool that is not `think` breaks
                                        # the streak, which is what makes the
                                        # limit a *consecutive* one.
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

                                        # Check if subtask is done. A failed
                                        # handler does not complete anything.
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

                                        # Check if main task is complete. The
                                        # return is deferred until every tool
                                        # call in this response has run.
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

                            # Handle all regular tools together
                            # MCP tools are served by the MCP manager, not by
                            # tool_struct, which resolves against self.tools
                            # only. Split them out first or every MCP call
                            # raises ToolNotFoundError.
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

                                    # Add to memory
                                    self.agent.short_memory.add(
                                        role="Tool Executor",
                                        content=format_data_structure(
                                            tool_output
                                        ),
                                    )
                                    self._map_batch_results(
                                        regular_tool_calls,
                                        tool_output,
                                        turn_results,
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

                                    # Handle tool call summary if enabled
                                    if (
                                        self.agent.tool_call_summary
                                        is True
                                    ):
                                        temp_llm = (
                                            self.agent.temp_llm_instance_for_tool_summary()
                                        )
                                        tool_response = temp_llm.run(
                                            f"""
                                            Please analyze and summarize the following tool execution output in a clear and concise way. 
                                            Focus on the key information and insights that would be most relevant to the user's original request.
                                            If there are any errors or issues, highlight them prominently.
                                            
                                            Tool Output:
                                            {tool_output}
                                            """
                                        )
                                        self.agent.short_memory.add(
                                            role=self.agent.agent_name,
                                            content=tool_response,
                                        )

                                except Exception as e:
                                    # Fallback to tool_execution_retry if direct execution fails
                                    if self.agent.verbose:
                                        logger.warning(
                                            f"Direct tool execution failed, using retry mechanism: {e}"
                                        )
                                    self.agent.tool_execution_retry(
                                        regular_tool_calls,
                                        subtask_iterations,
                                    )
                                    self._map_batch_results(
                                        regular_tool_calls,
                                        f"tool execution failed: {e}",
                                        turn_results,
                                    )

                            self._flush_tool_results(
                                turn_calls, turn_results
                            )

                            if task_complete:
                                return self.agent._generate_final_summary(
                                    streaming_callback=streaming_callback,
                                    messages=self._transcript.messages,
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
                                try:
                                    tool_output = self.agent.tool_struct.execute_function_calls_from_api_response(
                                        response
                                    )

                                    # Add to memory
                                    self.agent.short_memory.add(
                                        role="Tool Executor",
                                        content=format_data_structure(
                                            tool_output
                                        ),
                                    )

                                    # Display tool execution results using formatter
                                    if self.agent.print_on:
                                        tool_display = f"Tool Output:\n{format_data_structure(tool_output)}"
                                        formatter.print_panel(
                                            tool_display,
                                            title="Tool Execution Results",
                                        )

                                    # Handle tool call summary if enabled
                                    if (
                                        self.agent.tool_call_summary
                                        is True
                                    ):
                                        temp_llm = (
                                            self.agent.temp_llm_instance_for_tool_summary()
                                        )
                                        tool_response = temp_llm.run(
                                            f"""
                                            Please analyze and summarize the following tool execution output in a clear and concise way. 
                                            Focus on the key information and insights that would be most relevant to the user's original request.
                                            If there are any errors or issues, highlight them prominently.
                                            
                                            Tool Output:
                                            {tool_output}
                                            """
                                        )
                                        self.agent.short_memory.add(
                                            role=self.agent.agent_name,
                                            content=tool_response,
                                        )

                                except Exception as e:
                                    # Fallback to tool_execution_retry if direct execution fails
                                    if self.agent.verbose:
                                        logger.warning(
                                            f"Direct tool execution failed, using retry mechanism: {e}"
                                        )
                                    self.agent.tool_execution_retry(
                                        response, subtask_iterations
                                    )

                            self._flush_tool_results(
                                turn_calls, turn_results
                            )

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
                            # Force action. The nudge goes into the real
                            # transcript, not just short_memory, or the model
                            # never sees it on the next request.
                            nudge = (
                                "You have called `think` "
                                f"{self.agent.think_call_count} times in a row "
                                "without acting. Stop analysing. Take concrete "
                                "action now using the available tools, and call "
                                "subtask_done when the work is finished."
                            )
                            self.agent.short_memory.add(
                                role="system", content=nudge
                            )
                            self._transcript.append_user(nudge)

                            # Reset so the nudge gets a fair chance to work
                            # before firing again on the very next iteration.
                            self.agent.think_call_count = 0

                    except Exception as e:
                        if self.agent.verbose:
                            logger.error(
                                f"Error in subtask execution loop: {e}"
                            )
                        # Record the failure in the conversation. Without this
                        # the next iteration rebuilds an identical prompt and
                        # the model repeats whatever just failed.
                        self.agent.short_memory.add(
                            role="Tool Executor",
                            content=(
                                f"ERROR: the previous step failed with "
                                f"{type(e).__name__}: {e}. Adjust your "
                                "approach before retrying."
                            ),
                        )

                if not subtask_done:
                    # A subtask that burned its whole iteration budget without
                    # finishing is recorded as failed, not left pending. Left
                    # pending it stays eligible, so the outer loop re-selects it
                    # and re-runs the same doomed budget up to
                    # MAX_SUBTASK_ITERATIONS times - 100 x 20 = 2000 LLM calls
                    # for one stuck subtask. Failing it terminates the run,
                    # cascades `skipped` to its dependents, and reports honestly.
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
                streaming_callback=streaming_callback
            )

        except Exception as error:
            self.agent._handle_run_error(error)

    def _create_plan_tool(
        self, task_description: str, steps: List[Dict], **kwargs
    ) -> str:
        """
        Create a detailed plan for task execution.

        This tool is used by the autonomous loop to break down a complex task into
        manageable subtasks with dependencies, priorities, and execution order.

        **Plan Structure:**
        Each step in the plan must contain:
        - step_id (str): Unique identifier for the subtask
        - description (str): Detailed description of what needs to be done
        - priority (str, optional): Priority level (e.g., "high", "medium", "low")
        - dependencies (List[str], optional): List of step_ids that must complete first

        **Plan Storage:**
        The plan is stored in:
        - self.agent.autonomous_subtasks: List of all subtasks with their details
        - self.agent.subtask_status: Dictionary mapping step_id to status ("pending", "completed", "failed")
        - self.agent.plan_created: Boolean flag indicating plan creation

        **Execution Order:**
        Subtasks are executed based on:
        1. Dependencies: Tasks with unmet dependencies are blocked
        2. Priority: Higher priority tasks are preferred when multiple are available
        3. Creation order: Used as tiebreaker

        Args:
            task_description (str): High-level description of the overall task to be completed.
                This provides context for the subtask planning.
            steps (List[Dict]): List of step dictionaries, each containing:
                - step_id (str): Unique identifier for the subtask (required)
                - description (str): What needs to be accomplished (required)
                - priority (str): Priority level, e.g., "high", "medium", "low" (optional)
                - dependencies (List[str]): List of step_ids that must complete first (optional)
            **kwargs: Additional arguments (currently unused, reserved for future use).

        Returns:
            str: Confirmation message indicating successful plan creation with the number
                of subtasks created. Format: "Plan created successfully with {n} subtasks"

        Note:
            - Called during the planning phase, and again at any point during
              execution when the model revises the plan.
            - The call is idempotent, not destructive. Steps are merged by
              ``step_id``: work that already finished keeps its status and
              summary, still-pending steps are updated in place, unmentioned
              steps that already finished are retained as history, and
              unmentioned steps that are still pending are dropped. Only a
              first call on an empty plan starts from scratch.
            - If verbose=True, plan creation is logged with step details

        Examples:
            >>> steps = [
            ...     {
            ...         "step_id": "step1",
            ...         "description": "Set up project structure",
            ...         "priority": "high",
            ...         "dependencies": []
            ...     },
            ...     {
            ...         "step_id": "step2",
            ...         "description": "Implement authentication",
            ...         "priority": "high",
            ...         "dependencies": ["step1"]
            ...     }
            ... ]
            >>> result = agent._create_plan_tool(
            ...     "Build a web application",
            ...     steps
            ... )
            >>> # Returns: "Plan created successfully with 2 subtasks"
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
        # A revision may reference work that already finished but is not being
        # restated, so those ids stay valid dependency targets.
        known_step_ids |= set(existing)

        for step in steps:
            step_id = step.get("step_id", "")

            # step_id values in `dependencies` are model-generated free text.
            # A typo or hallucinated id used to satisfy the dependency check
            # silently; now it is dropped, with a warning, so the plan stays
            # runnable instead of deadlocking on a reference to nothing.
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

        # Merge, preserving the order the plan already had and appending
        # genuinely new work at the end.
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

        # A revision reports what changed, not the whole plan, so the model
        # can see the effect of its edit.
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

        # Prefer the cache the dynamic loader already populated, so this does
        # not add a network call per turn.
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
        Run MCP tool calls through the agent's MCP manager.

        Failures become the tool's result rather than propagating, so the model
        sees what went wrong and the transcript keeps one result per call id.
        """
        for call in mcp_calls:
            name = call.get("function", {}).get("name", "unknown")
            if self.agent.print_on:
                self.agent._visualize_function_call(name, {})

            try:
                self.agent.mcp_tool_handling(
                    response=[call], current_loop=current_loop
                )
                outcome = f"{name} executed via MCP. See the tool output above."
            except Exception as error:
                outcome = _format_tool_error(name, error)
                if self.agent.verbose:
                    logger.error(outcome)

            self.agent.short_memory.add(
                role="Tool Executor", content=f"{name}: {outcome}"
            )
            results[call.get("id", "")] = outcome

    def _prewarm_tools_from_plan(
        self, task_description: str, steps: List[Dict]
    ) -> List[str]:
        """
        Load the tools a plan implies, before any subtask starts.

        With ``dynamic_tools`` the model otherwise discovers tools one subtask
        at a time, because ``get_execution_prompt`` deliberately scopes each
        turn to a single subtask - so it cannot know what later steps need and
        searches again for each one, at a full round-trip each.

        The plan is the best statement of what the whole run needs and it
        exists before any subtask starts, so it is used as the query here.
        This costs no extra turn: it happens inside the ``create_plan`` call
        that just succeeded. If it misses nothing breaks - the model can still
        search mid-run exactly as before.

        Returns:
            Names of tools newly loaded, for reporting back to the model.
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
            # Speculative, so it demands stronger relevance than an explicit
            # search. A whole plan as the query contains enough common words
            # ("task", "data", "current") to give unrelated tools a nonzero
            # score, which would load the catalog and undo the saving.
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

            # Only an actually completed dependency satisfies. A failed one
            # cannot produce the output its dependents were planned around,
            # and an unknown id (None) means the reference is unresolvable --
            # neither should unblock execution.
            if all(status == "completed" for status in statuses):
                return subtask

            # Dependencies that failed or were skipped can never complete, so
            # this subtask is unreachable. Mark it skipped rather than leaving
            # it pending forever, so the run terminates and the final summary
            # can report what was not attempted.
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
