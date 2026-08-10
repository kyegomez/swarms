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
from swarms.utils.formatter import formatter
from swarms.utils.index import exists, format_data_structure


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

            self.agent.short_memory.add(
                role=self.agent.user_name, content=task
            )

            # Reset autonomous loop state
            self.agent.autonomous_subtasks = []
            self.agent.current_subtask_index = 0
            self.agent.subtask_status = {}
            self.agent.plan_created = False
            self.agent.think_call_count = 0

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

            # The `think` tool is redundant when the model already reasons natively
            # via extended thinking (thinking_tokens). Drop it to avoid unnecessary
            # tool-call round-trips and a cluttered tool list.
            if self.agent.thinking_tokens is not None:
                planning_tools = [
                    t
                    for t in planning_tools
                    if t.get("function", {}).get("name") != "think"
                ]

            if self.agent.tools_list_dictionary is None:
                self.agent.tools_list_dictionary = []

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
                "create_plan": self._create_plan_tool,
                "think": self._think_tool,
                "subtask_done": self._subtask_done_tool,
                "complete_task": self.agent._complete_task_tool,
                "respond_to_user": lambda **kwargs: respond_to_user_tool(
                    self, **kwargs
                ),
                "create_file": lambda **kwargs: create_file_tool(
                    self, **kwargs
                ),
                "update_file": lambda **kwargs: update_file_tool(
                    self, **kwargs
                ),
                "read_file": lambda **kwargs: read_file_tool(
                    self, **kwargs
                ),
                "list_directory": lambda **kwargs: list_directory_tool(
                    self, **kwargs
                ),
                "delete_file": lambda **kwargs: delete_file_tool(
                    self, **kwargs
                ),
                "run_bash": lambda **kwargs: run_bash_tool(
                    self, **kwargs
                ),
                "grep": lambda **kwargs: grep_tool(self, **kwargs),
                "create_sub_agent": lambda **kwargs: create_sub_agent_tool(
                    self, **kwargs
                ),
                "assign_task": lambda **kwargs: assign_task_tool(
                    self, **kwargs
                ),
                "check_sub_agent_status": lambda **kwargs: check_sub_agent_status_tool(
                    self, **kwargs
                ),
                "cancel_sub_agent_tasks": lambda **kwargs: cancel_sub_agent_tasks_tool(
                    self, **kwargs
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
            self.agent.short_memory.add(
                role=self.agent.user_name, content=planning_prompt
            )

            plan_created = False
            planning_attempts = 0
            max_planning_attempts = MAX_PLANNING_ATTEMPTS

            while (
                not plan_created
                and planning_attempts < max_planning_attempts
            ):
                planning_attempts += 1
                try:
                    task_prompt = (
                        self.agent.short_memory.return_history_as_string()
                    )
                    response = self.agent.call_llm(
                        task=task_prompt,
                        img=img,
                        current_loop=0,
                        streaming_callback=streaming_callback,
                        *args,
                        **kwargs,
                    )

                    response = self.agent.parse_llm_output(response)
                    self.agent.short_memory.add(
                        role=self.agent.agent_name, content=response
                    )

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

            # Integrate user tools after planning phase
            if exists(self.agent.tools):
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
                    progress = f"{sum(1 for s in self.agent.autonomous_subtasks if s['status'] in ['completed', 'failed'])}/{len(self.agent.autonomous_subtasks)}"
                    formatter.print_panel(
                        f"Subtask: {subtask_id}\nDescription: {subtask_desc}\nPriority: {subtask_priority}\nProgress: {progress} subtasks completed",
                        title=f"Executing Subtask: {subtask_id}",
                    )

                # Subtask execution loop: thinking -> tool actions -> observation
                subtask_iterations = 0
                max_subtask_loops = MAX_SUBTASK_LOOPS
                subtask_done = False

                # Add the execution prompt ONCE before the inner loop so the model
                # doesn't see duplicate copies of it on subsequent iterations and
                # mistakenly conclude "this task has been run before."
                execution_prompt = get_execution_prompt(
                    subtask_id,
                    subtask_desc,
                    self.agent.autonomous_subtasks,
                )
                self.agent.short_memory.add(
                    role=self.agent.user_name,
                    content=execution_prompt,
                )

                while (
                    not subtask_done
                    and subtask_iterations < max_subtask_loops
                ):
                    subtask_iterations += 1
                    self.agent.think_call_count = (
                        0  # Reset for each subtask
                    )

                    try:
                        task_prompt = (
                            self.agent.short_memory.return_history_as_string()
                        )
                        response = self.agent.call_llm(
                            task=task_prompt,
                            img=img,
                            current_loop=subtask_iterations,
                            streaming_callback=streaming_callback,
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

                        # Handle tool calls
                        if isinstance(response, list):
                            regular_tool_calls = []

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
                                    arguments = json.loads(
                                        tool_call["function"][
                                            "arguments"
                                        ]
                                    )

                                    # Handle planning tools and handoff tool
                                    if (
                                        function_name
                                        in planning_tool_handlers
                                    ):
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

                                            result = self.agent._handoff_task_tool(
                                                handoffs=handoffs_list
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

                                            result = planning_tool_handlers[
                                                function_name
                                            ](
                                                **arguments
                                            )

                                        # Add result to memory
                                        self.agent.short_memory.add(
                                            role="Tool Executor",
                                            content=f"{function_name} result: {result}",
                                        )

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

                                        # Check if subtask is done
                                        if (
                                            function_name
                                            == "subtask_done"
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
                                                break

                                        # Check if main task is complete
                                        if (
                                            function_name
                                            == "complete_task"
                                        ):
                                            # Task is complete, exit all loops
                                            return self.agent._generate_final_summary(
                                                streaming_callback=streaming_callback
                                            )
                                    else:
                                        # Collect regular tool calls for batch visualization and execution
                                        regular_tool_calls.append(
                                            tool_call
                                        )

                            # Handle all regular tools together
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
                            # Force action by adding a prompt
                            self.agent.short_memory.add(
                                role="system",
                                content="You have been thinking too much. Take action now using available tools.",
                            )

                    except Exception as e:
                        if self.agent.verbose:
                            logger.error(
                                f"Error in subtask execution loop: {e}"
                            )
                        # Continue to next iteration

                if not subtask_done:
                    if self.agent.print_on:
                        formatter.print_panel(
                            f"Subtask {subtask_id} not completed after {max_subtask_loops} iterations",
                            title="Subtask Timeout",
                        )
                    if self.agent.verbose:
                        logger.warning(
                            f"Subtask {subtask_id} not completed after {max_subtask_loops} iterations"
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
            - This method is called automatically by the autonomous loop during planning phase
            - The plan replaces any existing autonomous_subtasks
            - All subtasks start with status "pending"
            - current_subtask_index is reset to 0
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

        # Store the plan
        self.agent.autonomous_subtasks = []
        for step in steps:
            subtask = {
                "step_id": step.get("step_id", ""),
                "description": step.get("description", ""),
                "priority": step.get("priority", "medium"),
                "dependencies": step.get("dependencies", []),
                "status": "pending",
            }
            self.agent.autonomous_subtasks.append(subtask)
            self.agent.subtask_status[subtask["step_id"]] = "pending"

        self.agent.plan_created = True
        self.agent.current_subtask_index = 0

        if self.agent.verbose:
            logger.info(
                f"Plan created with {len(steps)} steps: {[s['step_id'] for s in self.agent.autonomous_subtasks]}"
            )
        return f"Plan created successfully with {len(steps)} subtasks"

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
            if subtask["status"] == "pending":
                # Check if all dependencies are completed
                dependencies = subtask.get("dependencies", [])
                if not dependencies or all(
                    self.agent.subtask_status.get(dep, "completed")
                    in ["completed", "failed"]
                    for dep in dependencies
                ):
                    return subtask

        return None

    def _all_subtasks_complete(self) -> bool:
        """
        Check if all subtasks are completed.

        Returns:
            bool: True if all subtasks are completed or failed
        """
        if not self.agent.autonomous_subtasks:
            return False

        return all(
            subtask["status"] in ["completed", "failed"]
            for subtask in self.agent.autonomous_subtasks
        )
