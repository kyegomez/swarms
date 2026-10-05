import json
import time
import traceback
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel

from swarms.prompts.handoffs_prompt import get_handoffs_prompt
from swarms.schemas.agent_errors import AgentToolExecutionError
from swarms.structs.autonomous_loop_utils import (
    get_autonomous_loop_tool_names,
)
from swarms.structs.transcript import Transcript
from swarms.tools.base_tool import BaseTool
from swarms.tools.dynamic_tool_loader import (
    DYNAMIC_TOOLS_NOTICE,
    SEARCH_TOOL_NAME,
    DynamicToolLoader,
)
from swarms.tools.handoffs_tool import handoff_task
from swarms.tools.handoffs_tool_schema import get_handoff_tool_schema
from swarms.tools.py_func_to_openai_func_str import (
    convert_multiple_functions_to_openai_function_schema,
)
from swarms.utils.formatter import formatter
from swarms.utils.index import exists, format_data_structure
from swarms.utils.litellm_wrapper import LiteLLM

TOOL_SUMMARY_PROMPT = """
Please analyze and summarize the following tool execution output in a clear and concise way.
Focus on the key information and insights that would be most relevant to the user's original request.
If there are any errors or issues, highlight them prominently.

Tool Output:
{output}
"""


def _name(tool_call: Any) -> Optional[str]:
    """Function name of a tool call dict, else None."""
    if isinstance(tool_call, dict):
        return tool_call.get("function", {}).get("name")
    return None


def _arguments(tool_call: dict) -> dict:
    """Tool call arguments as a dict; raises on malformed JSON."""
    arguments = tool_call.get("function", {}).get("arguments") or {}
    if isinstance(arguments, str):
        return json.loads(arguments)
    return arguments


def _clip(value: Any, limit: int) -> str:
    """Text of a value, cut to limit characters."""
    text = str(value)
    return text[:limit] + "..." if len(text) > limit else text


class ToolManager:
    """
    Runs tool setup, tool calls and tool retries for an agent.

    Args:
        agent: The owning Agent. Its tool state is read and written here.
    """

    def __init__(self, agent: Any):
        self.agent = agent

    def setup_tools(self) -> BaseTool:
        """
        Build the tool executor and add any pydantic tool schemas to memory.

        Returns:
            BaseTool: The executor, also stored on agent.tool_struct.
        """
        agent = self.agent
        agent.tool_struct = BaseTool(
            tools=agent.tools, verbose=agent.verbose
        )
        if exists(agent.tool_schema) or exists(
            agent.list_base_models
        ):
            self.handle_tool_schema_ops()
        return agent.tool_struct

    def handle_tool_schema_ops(self) -> None:
        """
        Add the agent's pydantic tool schemas to its memory.
        """
        agent = self.agent

        if exists(agent.tool_schema):
            logger.info(f"Tool schema provided: {agent.tool_schema}")
            agent.short_memory.add(
                role=agent.agent_name,
                content=agent.tool_struct.base_model_to_dict(
                    agent.tool_schema, output_str=True
                ),
            )

        if exists(agent.list_base_models):
            logger.info(
                "Multiple base models provided, Automatically converting to OpenAI function"
            )
            agent.short_memory.add(
                role=agent.agent_name,
                content=agent.tool_struct.multi_base_models_to_dict(
                    output_str=True
                ),
            )

    def load_tools(self) -> None:
        """
        Register handoffs, then load the tools eagerly or behind tool search.
        """
        agent = self.agent

        if exists(agent.handoffs):
            if agent.tools_list_dictionary is None:
                agent.tools_list_dictionary = []
            agent.tools_list_dictionary.extend(
                get_handoff_tool_schema()
            )

            registry = self.get_agent_registry()
            if registry:
                agent.system_prompt += "\n\n" + get_handoffs_prompt(
                    list(registry.values())
                )

        # Not exists(): exists([]) is True, so tools=[] deferred.
        if agent.dynamic_tools and (
            bool(agent.tools)
            or agent.mcp_enabled
            or agent.max_loops == "auto"
        ):
            agent.system_prompt += DYNAMIC_TOOLS_NOTICE
            self.setup_dynamic_tools()
        elif agent.tools:
            self.tool_handling()

    def tool_handling(self) -> None:
        """
        Add the agent's callable tools to its schema list and memory.
        """
        agent = self.agent
        if agent.tools_list_dictionary is None:
            agent.tools_list_dictionary = []

        seen = {
            schema["function"].get("name", "")
            for schema in agent.tools_list_dictionary
            if isinstance(schema, dict) and "function" in schema
        }
        for (
            schema
        ) in convert_multiple_functions_to_openai_function_schema(
            agent.tools
        ):
            name = schema.get("function", {}).get("name", "")
            if name not in seen:
                seen.add(name)
                agent.tools_list_dictionary.append(schema)

        agent.short_memory.add(
            role=agent.agent_name, content=agent.tools_list_dictionary
        )

    def get_agent_registry(self) -> Dict[str, Any]:
        """
        Map handoff agent names to agents.

        Returns:
            Dict[str, Any]: The registry built from agent.handoffs.
        """
        handoffs = self.agent.handoffs
        if isinstance(handoffs, dict):
            return handoffs
        if isinstance(handoffs, (list, tuple)):
            return {
                getattr(target, "agent_name", str(target)): target
                for target in handoffs
            }
        return {}

    def handoff_task_tool(
        self, handoffs: List[Dict[str, str]]
    ) -> str:
        """
        Delegate tasks to handoff agents and combine their responses.

        Args:
            handoffs: Requests with agent_name, task and reasoning keys.

        Returns:
            str: The aggregated responses.
        """
        return handoff_task(
            handoffs=handoffs,
            agent_registry=self.get_agent_registry(),
        )

    def add_mcp_tools_to_memory(self) -> List[Dict[str, Any]]:
        """
        Fetch the tool schemas exposed by the configured MCP servers.

        Returns:
            List[Dict[str, Any]]: OpenAI tool schemas from every server.

        Raises:
            AgentMCPConnectionError: If no server could be reached.
        """
        agent = self.agent
        try:
            tools = agent.mcp_manager.get_tools()
        except Exception as e:
            logger.error(
                f"Error Adding MCP Tools to Agent: {agent.agent_name} Error: {e} Traceback: {traceback.format_exc()}"
            )
            raise

        if agent.print_on:
            agent.pretty_print(
                f"✨ [SYSTEM] Successfully integrated {len(tools)} MCP tools into agent: {agent.agent_name} | Status: ONLINE | Time: {time.strftime('%H:%M:%S')} ✨",
                loop_count=0,
            )
        return tools

    def setup_dynamic_tools(
        self, always_loaded: Optional[List[dict]] = None
    ) -> DynamicToolLoader:
        """
        Defer the agent's tool schemas behind a tool_search tool.

        Args:
            always_loaded: Schemas never deferred, such as control-flow tools.

        Returns:
            DynamicToolLoader: The loader, also stored on agent.tool_loader.
        """
        agent = self.agent
        loader = DynamicToolLoader(tools=agent.tools or [])

        # Keep registered handoff and MCP schemas; drop deferred user tools and tool_search, which schemas() re-adds.
        dropped = set(loader.deferred_names) | {SEARCH_TOOL_NAME}
        preserved = [
            schema
            for schema in agent.tools_list_dictionary or []
            if _name(schema) not in dropped
        ]
        keep, seen = [], set()
        for schema in list(always_loaded or []) + preserved:
            name = _name(schema)
            if name and name not in seen:
                seen.add(name)
                keep.append(schema)
        loader.always_loaded = keep

        # A rebuilt loader is empty; the fetch guard will not refetch.
        for schema in agent._mcp_schemas_cache or []:
            loader.register_schema(schema)

        agent.tool_loader = loader
        agent.tools_list_dictionary = loader.schemas()
        return loader

    def defer_tool_schemas(self, schemas: List[dict]) -> None:
        """
        Add pre-built schemas to the deferred catalog.

        Args:
            schemas: OpenAI tool schemas, such as MCP or loop tools.
        """
        agent = self.agent
        if agent.tool_loader is None:
            return
        for schema in schemas:
            agent.tool_loader.register_schema(schema)
        agent.tools_list_dictionary = agent.tool_loader.schemas()

    def defer_mcp_tools(self) -> int:
        """
        Move the MCP tool schemas into the deferred catalog, fetching them once.

        Returns:
            int: Schemas added; 0 if MCP or dynamic tools are off or nothing is new.
        """
        agent = self.agent
        if agent.tool_loader is None or not agent.mcp_enabled:
            return 0

        # Keyed on loader contents, not a flag: the autonomous loop builds a fresh loader per run.
        cached = agent._mcp_schemas_cache
        if cached is not None and all(
            _name(schema) in agent.tool_loader for schema in cached
        ):
            return 0

        if cached is None:
            try:
                cached = self.add_mcp_tools_to_memory()
            except Exception as error:
                # An unreachable server must not take down agent setup.
                logger.error(
                    f"Could not fetch MCP tools to defer: {error}"
                )
                agent._mcp_schemas_cache = []
                agent._mcp_tools_deferred = True
                return 0
            agent._mcp_schemas_cache = cached

        agent._mcp_tools_deferred = True
        self.defer_tool_schemas(cached)
        if agent.verbose:
            logger.info(
                f"Deferred {len(cached)} MCP tool(s) into the catalog: "
                f"{[_name(schema) for schema in cached]}"
            )
        return len(cached)

    def list_tools(self) -> List[str]:
        """
        Names of every tool the agent can call.

        Local tools from ``tools=`` and ``tools_list_dictionary=`` come
        first, then tools from MCP servers, then the built-ins:
        ``handoff_task``, ``tool_search`` and the autonomous-loop tools.
        Tools deferred behind ``tool_search`` are included, since the
        agent can load and call them. MCP schemas are fetched at most
        once per agent; if no server can be reached the error is logged
        and the other names are still returned.

        Returns:
            List[str]: Tool names, without duplicates, in a stable order.
        """
        agent = self.agent
        builtins = []
        if exists(agent.handoffs):
            builtins.extend(
                _name(schema) for schema in get_handoff_tool_schema()
            )
        if agent.tool_loader is not None:
            builtins.append(SEARCH_TOOL_NAME)
        if agent.max_loops == "auto":
            loop_names = get_autonomous_loop_tool_names()
            if (
                agent.selected_tools != "all"
                and agent.selected_tools is not None
            ):
                loop_names = [
                    name
                    for name in loop_names
                    if name in agent.selected_tools
                ]
            if not getattr(agent, "think_tool", False):
                loop_names = [
                    name for name in loop_names if name != "think"
                ]
            builtins.extend(loop_names)

        mcp_schemas = []
        if agent.mcp_enabled:
            if agent._mcp_schemas_cache is None:
                try:
                    agent._mcp_schemas_cache = (
                        agent.mcp_manager.get_tools()
                    )
                except Exception as error:
                    logger.error(
                        f"Could not list MCP tools for {agent.agent_name}: {error}"
                    )
            mcp_schemas = agent._mcp_schemas_cache or []
        mcp_names = [_name(schema) for schema in mcp_schemas]

        local_names = [
            _name(schema)
            for schema in agent.tools_list_dictionary or []
        ]
        if agent.tool_loader is not None:
            local_names.extend(agent.tool_loader.loaded_names)
            local_names.extend(agent.tool_loader.deferred_names)

        reserved = set(mcp_names) | set(builtins)
        ordered = [
            name for name in local_names if name not in reserved
        ]
        return [
            name
            for name in dict.fromkeys(ordered + mcp_names + builtins)
            if name
        ]

    def tool_search_tool(
        self,
        query: str,
        max_results: int = 5,
        min_score_ratio: float = 0.0,
        **kwargs,
    ) -> str:
        """
        Search the deferred catalog and load the matching tools.

        Args:
            query: What the model is looking for.
            max_results: Most tools to load.
            min_score_ratio: Minimum score relative to the best match.

        Returns:
            str: The search result shown to the model.
        """
        agent = self.agent
        if agent.tool_loader is None:
            return (
                "Tool search is unavailable: this agent was not built with "
                "dynamic_tools=True."
            )

        result = agent.tool_loader.run_search(
            query=query,
            max_results=max_results,
            min_score_ratio=min_score_ratio,
        )
        agent.tools_list_dictionary = agent.tool_loader.schemas()
        # Rebuilt so the newly loaded schemas are sent on the next request.
        if agent.llm is not None:
            agent.llm = agent.llm_handling()

        if agent.verbose:
            logger.info(
                f"tool_search({query!r}) -> loaded {agent.tool_loader.loaded_names}"
            )
        return result

    def parse_llm_output(self, response: Any) -> Any:
        """
        Normalize a model response to text or a list of tool call dicts.

        Args:
            response: The response from the model in any format.

        Returns:
            Any: The response text, or a list of tool call dicts.

        Raises:
            ValueError: If the response cannot be normalized.
        """
        try:
            if isinstance(response, dict):
                if "choices" in response:
                    return response["choices"][0]["message"][
                        "content"
                    ]
                # MCP returns a bare dict for one call and a list for several; normalise so isinstance(list) holds.
                if "function" in response:
                    return [response]
                return json.dumps(response)
            if isinstance(response, BaseModel):
                return response.model_dump()
            if (
                isinstance(response, list)
                and response
                and isinstance(response[0], BaseModel)
            ):
                return [item.model_dump() for item in response]
            return response
        except Exception as e:
            logger.error(f"Error parsing LLM output: {e}")
            raise ValueError(
                f"Failed to parse LLM output: {type(response)}"
            ) from e

    def parse_response(self, response: Any) -> Any:
        """
        Normalize one main-loop model response before it is recorded.

        Args:
            response: The raw response from the model call.

        Returns:
            Any: The response text, or a list of tool call dicts.
        """
        if exists(self.agent.tools_list_dictionary) and isinstance(
            response, BaseModel
        ):
            response = response.model_dump()
        return self.parse_llm_output(response)

    def handle_tool_calls(
        self,
        response: Any,
        loop_count: int,
        transcript: Optional[Transcript] = None,
        turn_calls: Optional[list] = None,
        turn_results: Optional[dict] = None,
    ) -> None:
        """
        Run every tool call in one model response and answer each in the transcript.

        Args:
            response: The parsed model response.
            loop_count: The current loop number.
            transcript: The run's transcript, or None when transforms are on.
            turn_calls: Tool calls recorded in the transcript this turn.
            turn_results: Results keyed by tool call id, updated in place.

        Raises:
            AgentToolExecutionError: If the callable tools fail every retry.
        """
        agent = self.agent
        turn_results = {} if turn_results is None else turn_results

        if isinstance(response, list):
            if agent.tool_loader:
                response = self._run_tool_search_calls(
                    response, turn_results
                )
                # Falling through with nothing left would log a misleading "no function calls found".
                if not response:
                    return self._flush(
                        transcript, turn_calls, turn_results
                    )
            self._run_handoff_calls(
                response, turn_results, loop_count
            )

        if exists(agent.tools):
            output = self.tool_execution_retry(response, loop_count)
            if transcript is not None and turn_calls:
                transcript.map_batch_results(
                    [{"id": call["id"]} for call in turn_calls],
                    output,
                    turn_results,
                    formatter=format_data_structure,
                )

        if agent.mcp_enabled:
            if response is None:
                logger.warning(
                    f"LLM returned None response in loop {loop_count}, skipping MCP tool handling"
                )
            else:
                self.mcp_tool_handling(
                    response=response, current_loop=loop_count
                )

        self._flush(transcript, turn_calls, turn_results)

    def _flush(
        self,
        transcript: Optional[Transcript],
        turn_calls: Optional[list],
        turn_results: dict,
    ) -> None:
        """Answer every recorded tool call; a gap makes the next request invalid."""
        if transcript is not None and turn_calls:
            transcript.flush_tool_results(turn_calls, turn_results)

    def _run_tool_search_calls(
        self, response: list, turn_results: dict
    ) -> list:
        """Run the tool_search calls in a response and return the rest."""
        agent = self.agent
        for call in response:
            if _name(call) != SEARCH_TOOL_NAME:
                continue
            try:
                arguments = _arguments(call)
            except (ValueError, TypeError):
                arguments = {}

            result = self.tool_search_tool(**arguments)
            agent.short_memory.add(
                role="Tool Executor",
                content=f"tool_search result: {result}",
            )
            turn_results[call.get("id", "")] = result
            if agent.print_on:
                formatter.print_panel(result, title="Tool Search")

        return [
            call
            for call in response
            if _name(call) != SEARCH_TOOL_NAME
        ]

    def _run_handoff_calls(
        self, response: list, turn_results: dict, loop_count: int
    ) -> None:
        """Run the handoff_task calls in a response."""
        agent = self.agent
        for call in response:
            if _name(call) != "handoff_task":
                continue
            handoffs = _arguments(call).get("handoffs", [])
            self.visualize_handoff_call(handoffs, call)

            result = self.handoff_task_tool(handoffs=handoffs)
            agent.short_memory.add(
                role="Tool Executor",
                content=f"Handoff Result:\n{result}",
            )
            turn_results[call.get("id", "")] = result

            if agent.print_on:
                delegated = ", ".join(
                    target.get("agent_name", "<unknown>")
                    for target in handoffs
                )
                agent.pretty_print(
                    f"[Handoff] Delegated tasks to {len(handoffs)} agent(s): {delegated}\nSuccessfully executed handoff_task function.",
                    loop_count,
                )

    def handle_complete_task(self, response: Any) -> Optional[str]:
        """
        Run the first complete_task call in a final summary response.

        Args:
            response: The parsed final summary response.

        Returns:
            Optional[str]: The completion summary, or None if complete_task was not called.
        """
        if not isinstance(response, list):
            return None

        for call in response:
            if _name(call) != "complete_task":
                continue
            arguments = _arguments(call)
            self.visualize_function_call("complete_task", arguments)

            result = self.complete_task_tool(**arguments)
            self.agent.short_memory.add(
                role="Tool Executor",
                content=f"complete_task result: {result}",
            )
            if self.agent.print_on:
                formatter.print_panel(
                    result, title="Task Completion Summary"
                )
            return result

        return None

    def complete_task_tool(
        self,
        task_id: str,
        summary: str,
        success: bool,
        results: Optional[str] = None,
        lessons_learned: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Mark the main task complete and record a summary of every subtask.

        Args:
            task_id: Identifier of the main task.
            summary: Summary of the whole task.
            success: Whether the task succeeded.
            results: Detailed results, if any.
            lessons_learned: Insights from the run, if any.

        Returns:
            str: The completion summary, also added to memory.
        """
        agent = self.agent
        if agent.verbose:
            logger.info(f"Completing main task {task_id}: {summary}")
            incomplete = [
                s["step_id"]
                for s in agent.autonomous_subtasks
                if s["status"] not in ["completed", "failed"]
            ]
            if incomplete:
                logger.warning(
                    f"Attempting to complete task but {len(incomplete)} subtasks are not done: {incomplete}"
                )

        text = (
            f"Task Completion Summary\n\nTask ID: {task_id}\n"
            f"Status: {'Success' if success else 'Failed'}\n"
            f"Summary: {summary}\n"
        )
        if results:
            text += f"\nResults:\n{results}\n"
        if lessons_learned:
            text += f"\nLessons Learned:\n{lessons_learned}\n"

        text += "\nSubtask Breakdown:\n"
        for subtask in agent.autonomous_subtasks:
            text += f"- {subtask['step_id']}: {subtask.get('status', 'unknown')} - {subtask.get('description', '')}\n"
            if "summary" in subtask:
                text += f"  Summary: {subtask['summary']}\n"

        agent.short_memory.add(role=agent.agent_name, content=text)
        if lessons_learned and agent.persistent_memory:
            agent.short_memory.record_lesson(
                lesson=lessons_learned,
                task=summary,
                outcome=success,
            )
        if agent.verbose:
            logger.info(
                "Main task marked as completed with comprehensive summary"
            )
        return text

    def execute_tools(self, response: Any, loop_count: int) -> None:
        """
        Execute the callable tool calls in a response and record their output.

        Args:
            response: A tool call dict or a list of them.
            loop_count: The current loop number, used when printing.

        Raises:
            Exception: Whatever the tool executor raises.
        """
        agent = self.agent
        if response is None:
            logger.warning(
                f"Cannot execute tools with None response in loop {loop_count}. "
                "This may indicate the LLM did not return a valid response."
            )
            return

        calls = [
            call
            for call in (
                response if isinstance(response, list) else [response]
            )
            if isinstance(call, dict)
        ]
        if agent.print_on:
            for call in calls:
                try:
                    arguments = _arguments(call)
                except (ValueError, TypeError, AttributeError):
                    arguments = {}
                self.visualize_function_call(
                    _name(call) or "Unknown",
                    arguments if isinstance(arguments, dict) else {},
                    call_id=call.get("id"),
                )

        output = agent.tool_struct.execute_function_calls_from_api_response(
            response
        )
        # Stored so a transcript builder can map it to tool_call ids.
        agent._last_tool_output = output

        # A reply with no tool calls parses to []; recording it would bury the answer under "[] (empty list)".
        if not output:
            return

        formatted = format_data_structure(output)
        agent.short_memory.add(
            role="Tool Executor", content=formatted
        )

        if agent.print_on:
            stamp = time.strftime("%H:%M:%S")
            if agent.show_tool_execution_output is True:
                executed = "".join(
                    f"  - {_name(call) or 'Unknown'}"
                    + (
                        f" (ID: {call['id']})"
                        if call.get("id")
                        else ""
                    )
                    + f" [{call.get('type', 'function')}]\n"
                    for call in calls
                )
                header = (
                    f"Tools Executed:\n{executed}\n"
                    if executed
                    else ""
                )
                formatter.print_panel(
                    f"Execution Time: {stamp}\n\n{header}Output:\n{formatted}",
                    title="Tool Execution Results",
                )
            else:
                names = ", ".join(
                    _name(call) or "Unknown" for call in calls
                )
                formatter.print_panel(
                    (
                        f"Tools Executed: {names}\nTime: {stamp}"
                        if names
                        else f"Tool Executed Successfully [{stamp}]"
                    ),
                    title="Tool Execution",
                )

        # A temporary LLM instead of mutating the cached one.
        if agent.tool_call_summary is True:
            summary = self.temp_llm_instance_for_tool_summary().run(
                TOOL_SUMMARY_PROMPT.format(output=output)
            )
            agent.short_memory.add(
                role=agent.agent_name, content=summary
            )
            if agent.print_on is True:
                agent.pretty_print(summary, loop_count)

    def tool_execution_retry(
        self, response: Any, loop_count: int
    ) -> Any:
        """
        Execute tools, retrying up to agent.tool_retry_attempts times.

        Args:
            response: The model response holding the tool calls.
            loop_count: The current loop number.

        Returns:
            Any: The tool output, or None when the response is None.

        Raises:
            AgentToolExecutionError: If every attempt fails.
        """
        agent = self.agent
        if response is None:
            logger.warning(
                f"Agent '{agent.agent_name}' received None response from LLM in loop {loop_count}. "
                f"This may indicate an issue with the model or prompt. Skipping tool execution."
            )
            return None

        attempts = max(1, int(agent.tool_retry_attempts or 1))
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                self.execute_tools(
                    response=response, loop_count=loop_count
                )
                return getattr(agent, "_last_tool_output", None)
            except Exception as e:
                last_error = e
                logger.error(
                    f"Agent '{agent.agent_name}' tool execution failed on attempt "
                    f"{attempt}/{attempts} in loop {loop_count}: {str(e)}. "
                    f"Full traceback: {traceback.format_exc()}"
                )

        # Attempts exhausted: raise, or the model reads a silent no-op as success.
        raise AgentToolExecutionError(
            f"Agent '{agent.agent_name}' failed to execute tools in loop "
            f"{loop_count} after {attempts} attempt(s): {last_error}"
        ) from last_error

    def mcp_tool_handling(
        self, response: Any, current_loop: Optional[int] = 0
    ) -> None:
        """
        Execute the MCP tool calls in a response and record a summary.

        Args:
            response: The model response holding MCP tool calls.
            current_loop: The current loop number, used when printing.

        Raises:
            AgentMCPConnectionError: If no MCP server could be reached.
            AgentMCPToolError: If tool execution fails outright.
        """
        agent = self.agent
        try:
            tool_response = agent.mcp_manager.execute_tool_calls(
                response, output_type="dict"
            )
            if not tool_response:
                if agent.verbose:
                    logger.info(
                        f"No MCP tool calls found in the response for {agent.agent_name}"
                    )
                return

            text = f"MCP Tool Response: \n\n {json.dumps(tool_response, indent=2, default=str)}"
            if agent.print_on is True:
                formatter.print_panel(
                    content=text,
                    title="MCP Tool Response: 🛠️",
                    style="green",
                )
            agent.short_memory.add(role="Tool Executor", content=text)

            try:
                summary = (
                    self.temp_llm_instance_for_tool_summary().run(
                        task=agent.short_memory.get_str()
                    )
                )
            except Exception as e:
                logger.error(
                    f"Error calling LLM after MCP tool execution: {e}"
                )
                summary = "I successfully executed the MCP tool and retrieved the information above."

            if agent.print_on is True:
                agent.pretty_print(summary, loop_count=current_loop)
            agent.short_memory.add(
                role=agent.agent_name, content=summary
            )
        except Exception as e:
            logger.error(
                f"Error in MCP tool handling for {agent.agent_name}: {e} Traceback: {traceback.format_exc()}"
            )
            raise

    def temp_llm_instance_for_tool_summary(self) -> LiteLLM:
        """
        Build a tool-free, non-streaming model for summarizing tool output.

        Returns:
            LiteLLM: A fresh model client with the agent's settings.
        """
        agent = self.agent
        return LiteLLM(
            model_name=agent.model_name,
            temperature=agent.temperature,
            top_p=agent.top_p,  # Anthropic rejects requests with both temperature and top_p
            max_tokens=agent.max_tokens,
            system_prompt=agent.system_prompt + agent._skills_prompt,
            stream=False,
            tools_list_dictionary=None,
            parallel_tool_calls=False,
            base_url=agent.llm_base_url,
            api_key=agent.llm_api_key,
            usage_hook=agent._add_usage,
        )

    def visualize_function_call(
        self,
        function_name: str,
        arguments: Dict[str, Any],
        result: Any = None,
        call_id: Optional[str] = None,
    ) -> None:
        """
        Print a function call panel when printing is on.

        Args:
            function_name: Name of the function being called.
            arguments: Arguments passed to the function.
            result: The function's result, if it has run.
            call_id: The tool call id, if any.
        """
        agent = self.agent
        if not agent.print_on:
            return

        content = f"Function: {function_name}\n"
        if call_id:
            content += f"Call ID: {call_id}\n"
        content += "\nArguments:\n" + "".join(
            f"  {key}: {_clip(value, 200)}\n"
            for key, value in arguments.items()
        )
        if result:
            content += f"\nResult:\n{_clip(result, 500)}"

        formatter.print_panel(
            content,
            title=f"Agent: {agent.agent_name} Function Call: {function_name}",
        )

    def visualize_handoff_call(
        self,
        handoffs: List[Dict[str, str]],
        tool_call: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Print a handoff panel listing every delegation when printing is on.

        Args:
            handoffs: Requests with agent_name, task and reasoning keys.
            tool_call: The handoff tool call, used for its id.
        """
        agent = self.agent
        if not agent.print_on:
            return

        content = f"Function: handoff_task\nDelegating to {len(handoffs)} agent(s)\n\n"
        if tool_call and tool_call.get("id"):
            content += f"Call ID: {tool_call.get('id')}\n\n"
        content += "Handoff Details:\n" + "=" * 80 + "\n"

        for i, handoff in enumerate(handoffs, 1):
            content += (
                f"\n[{i}] Agent: {handoff.get('agent_name', '<unknown>')}\n"
                f"    Task: {_clip(handoff.get('task', ''), 150)}\n"
                f"    Reasoning: {_clip(handoff.get('reasoning', ''), 150)}\n"
            )
            if i < len(handoffs):
                content += "\n" + "-" * 80 + "\n"

        formatter.print_panel(
            content,
            title=f"Agent: {agent.agent_name} Handoff Tool Call",
        )
