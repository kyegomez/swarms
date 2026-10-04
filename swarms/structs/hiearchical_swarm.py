import asyncio
import inspect
import json
import traceback
from concurrent.futures import as_completed
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger

from swarms.prompts.hierarchical_swarm_prompts import (
    AGENT_TASK_TEMPLATE,
    DIRECTOR_FEEDBACK_PROMPT,
    DIRECTOR_PLANNING_PROMPT,
    HIEARCHICAL_SWARM_SYSTEM_PROMPT,
    HIERARCHICAL_SWARM_JUDGE_PROMPT,
    LOOP_CONTINUATION_PROMPT,
    WORKER_RECOVERY_PROMPT,
)
from swarms.prompts.multi_agent_collab_prompt import (
    MULTI_AGENT_COLLAB_PROMPT_TWO,
)
from swarms.schemas.hs_schemas import (
    HierarchicalOrder,
    JudgeReport,
    OrderBatch,
    SwarmSpec as SwarmSpec,
)
from swarms.structs.agent import Agent
from swarms.structs.context_utils import (
    messages_for,
    new_context_for,
    split_last_turn,
)
from swarms.structs.conversation import Conversation
from swarms.structs.execution_utils import batched_run
from swarms.structs.hierarchical_order_parser import (
    parse_orders as _parse_orders,
)
from swarms.structs.ma_utils import list_all_agents
from swarms.structs.omni_agent_types import AgentListType
from swarms.telemetry.otel import (
    ContextThreadPoolExecutor,
    capture_init,
    trace_run,
)
from swarms.tools.base_tool import BaseTool
from swarms.utils.any_to_str import any_to_str
from swarms.utils.formatter import formatter
from swarms.utils.get_cpu_cores import max_workers_95_percent
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.output_types import OutputType
from swarms.utils.workspace_manager import WorkspaceManager

_ORDER_BATCH_SCHEMA = BaseTool().base_model_to_dict(OrderBatch)
_JUDGE_REPORT_SCHEMA = BaseTool().base_model_to_dict(JudgeReport)


async def _arun_agent(agent: Any, *args, **kwargs) -> Any:
    """Await ``agent.arun`` when it is async, else run ``agent.run`` in a thread."""
    arun = getattr(agent, "arun", None)
    if inspect.iscoroutinefunction(arun):
        return await arun(*args, **kwargs)
    return await asyncio.to_thread(agent.run, *args, **kwargs)


class HierarchicalSwarm:
    """Coordinate a director and workers across iterative task loops."""

    def __init__(
        self,
        name: str = "HierarchicalAgentSwarm",
        description: str = "Distributed task swarm",
        director: Optional[Union[Agent, Callable, Any]] = None,
        agents: AgentListType = None,
        max_loops: int = 1,
        output_type: OutputType = "dict-all-except-first",
        feedback_director_model_name: str = "gpt-5.4",
        director_name: str = "Director",
        director_model_name: str = "gpt-5.4",
        add_collaboration_prompt: bool = True,
        director_feedback_on: bool = False,
        interactive: bool = False,
        director_system_prompt: str = HIEARCHICAL_SWARM_SYSTEM_PROMPT,
        multi_agent_prompt_improvements: bool = False,
        director_temperature: float = 0.7,
        director_top_p: float = 0.9,
        planning_enabled: bool = False,
        autosave: bool = False,
        verbose: bool = False,
        print_on: bool = True,
        parallel_execution: bool = True,
        max_workers: Optional[int] = None,
        agent_as_judge: bool = False,
        judge_agent_model_name: str = "gpt-5.4",
        director_settings: Optional[Dict[str, Any]] = None,
        max_agent_retries: int = 1,
        max_reassignment_attempts: int = 1,
        *args,
        **kwargs,
    ):
        """Initialize the swarm.

        Args:
            name: Swarm name.
            description: Swarm purpose.
            director: Director agent; created when omitted.
            agents: Worker agents or nested orchestrators.
            max_loops: Maximum orchestration loops.
            output_type: Result format.
            feedback_director_model_name: Feedback model.
            director_name: Director name.
            director_model_name: Director model.
            add_collaboration_prompt: Add worker collaboration context.
            director_feedback_on: Enable feedback loops.
            interactive: Prompt for a missing task.
            director_system_prompt: Director instructions.
            multi_agent_prompt_improvements: Enrich worker prompts.
            director_temperature: Director temperature.
            director_top_p: Director nucleus sampling.
            planning_enabled: Run a planning pass.
            autosave: Save conversation history.
            verbose: Enable verbose output.
            print_on: Print the director's plan and orders each step.
            parallel_execution: Execute orders concurrently.
            max_workers: Worker thread limit.
            agent_as_judge: Evaluate worker outputs.
            judge_agent_model_name: Judge model.
            director_settings: Additional director settings.
            max_agent_retries: Retries per failed order.
            max_reassignment_attempts: Recovery attempts.
            *args: Reserved positional arguments.
            **kwargs: Reserved keyword arguments.

        Raises:
            ValueError: If configuration is invalid.
        """
        self.name = name
        self.description = description
        self.director = director
        self.agents = agents
        self.max_loops = max_loops
        self.output_type = output_type
        self.feedback_director_model_name = (
            feedback_director_model_name
        )
        self.director_name = director_name
        self.director_model_name = director_model_name
        self.add_collaboration_prompt = add_collaboration_prompt
        self.director_feedback_on = director_feedback_on
        self.interactive = interactive
        self.director_system_prompt = director_system_prompt
        self.multi_agent_prompt_improvements = (
            multi_agent_prompt_improvements
        )
        self.director_temperature = director_temperature
        self.director_top_p = director_top_p
        self.director_settings = dict(director_settings or {})
        self.director_name = self.director_settings.get(
            "agent_name", self.director_name
        )
        self.director_model_name = self.director_settings.get(
            "model_name", self.director_model_name
        )
        self.director_system_prompt = self.director_settings.get(
            "system_prompt", self.director_system_prompt
        )
        self.director_temperature = self.director_settings.get(
            "temperature", self.director_temperature
        )
        self.director_top_p = self.director_settings.get(
            "top_p", self.director_top_p
        )
        self.max_agent_retries = max_agent_retries
        self.max_reassignment_attempts = max_reassignment_attempts
        self.planning_enabled = planning_enabled
        self.autosave = autosave
        self.verbose = verbose
        self.print_on = print_on
        self.parallel_execution = parallel_execution
        self.max_workers = (
            max_workers
            if max_workers is not None
            else max_workers_95_percent()
        )
        self.agent_as_judge = agent_as_judge
        self.judge_agent_model_name = judge_agent_model_name
        self._feedback_director = None
        self._judge_agent = None
        self._planning_director = None
        self.workspace = WorkspaceManager(
            self,
            name=self.name or "hierarchical-swarm",
            verbose=self.verbose,
            enabled=self.autosave,
        )
        self.swarm_workspace_dir = self.workspace.dir

        # How much of the shared conversation each agent has already seen.
        self._delivered: Dict[str, int] = {}

        self.conversation = Conversation(time_enabled=False)

        self.initialize_swarm()

        capture_init(self)

    def initialize_swarm(self):
        if self.interactive:
            self.agents_no_print()

        self.init_swarm()

    def list_worker_agents(self) -> str:
        return list_all_agents(
            agents=self.agents,
            add_to_conversation=False,
        )

    def display_hierarchy(self) -> None:
        """Print the director-worker hierarchy."""
        formatter.display_hierarchy(
            director_name=self.director_name,
            director_model_name=self.director_model_name,
            agents=self.agents,
            swarm_name=self.name,
        )

    def prepare_worker_agents(self):
        for agent in self.agents:
            prompt = (
                MULTI_AGENT_COLLAB_PROMPT_TWO
                + self.list_worker_agents()
            )
            if hasattr(agent, "system_prompt"):
                agent.system_prompt += prompt
            else:
                agent.system_prompt = prompt

    def init_swarm(self):
        """Initialize conversation state and validate the swarm."""
        # Reliability checks
        self.reliability_checks()

        # Hierarchical swarms pass only final responses between agents.
        self.enforce_final_agent_outputs()

        # Add agent context to the director
        self.add_context_to_director()

        if self.multi_agent_prompt_improvements:
            self.prepare_worker_agents()

    def enforce_final_agent_outputs(self) -> None:
        """Force every configurable agent to return only its final response."""
        agents = [self.director, *(self.agents or [])]
        for agent in agents:
            if hasattr(agent, "output_type"):
                agent.output_type = "final"

    def add_context_to_director(self):
        """Add the worker roster to shared context."""
        try:
            list_all_agents(
                agents=self.agents,
                conversation=self.conversation,
                add_to_conversation=True,
                add_collaboration_prompt=self.add_collaboration_prompt,
            )

        except Exception as e:
            logger.error(
                f"[ERROR] Failed to add context to director: {e} | Traceback: {traceback.format_exc()}"
            )

    def setup_director(self):
        """Create the structured-output director.

        Returns:
            Configured director agent.

        Raises:
            Exception: If director creation fails.
        """
        try:
            settings = {
                "agent_name": self.director_name,
                "agent_description": "A director agent that can create a plan and distribute orders to agents",
                "system_prompt": self.director_system_prompt,
                "model_name": self.director_model_name,
                "temperature": self.director_temperature,
                "top_p": self.director_top_p,
                "max_loops": 1,
                "base_model": OrderBatch,
                "tools_list_dictionary": [_ORDER_BATCH_SCHEMA],
                "output_type": "final",
            }
            settings.update(
                {
                    key: value
                    for key, value in self.director_settings.items()
                    if key != "planning_system_prompt"
                }
            )
            settings["output_type"] = "final"
            return Agent(**settings)

        except Exception as e:
            logger.error(
                f"[ERROR] Failed to setup director: {e} | Traceback: {traceback.format_exc()} | If this issue persists, please report it at: https://github.com/kyegomez/swarms/issues"
            )
            raise

    def _get_planning_director(self) -> Agent:
        """Cached planning director; built once per swarm instance."""
        if self._planning_director is None:
            settings = {
                "agent_name": self.director_name,
                "agent_description": "A director agent that can create a plan and distribute orders to agents",
                "model_name": self.director_model_name,
                "temperature": self.director_temperature,
                "top_p": self.director_top_p,
                "max_loops": 1,
                "output_type": "final",
            }
            settings.update(
                {
                    key: value
                    for key, value in self.director_settings.items()
                    if key
                    not in {
                        "base_model",
                        "tools_list_dictionary",
                        "planning_system_prompt",
                    }
                }
            )
            settings["system_prompt"] = self.director_settings.get(
                "planning_system_prompt", DIRECTOR_PLANNING_PROMPT
            )
            settings["output_type"] = "final"
            self._planning_director = Agent(**settings)
        return self._planning_director

    def setup_director_with_planning(
        self, task: str = None, img: Optional[str] = None
    ):
        return self._get_planning_director().run(task=task, img=img)

    def reliability_checks(self):
        if not self.agents or len(self.agents) == 0:
            raise ValueError(
                "No agents found in the swarm. At least one agent must be provided to create a hierarchical swarm."
            )

        if self.max_loops <= 0:
            raise ValueError(
                "Max loops must be greater than 0. Please set a valid number of loops."
            )

        if self.max_agent_retries < 0:
            raise ValueError(
                "max_agent_retries must be greater than or equal to 0."
            )

        if self.max_reassignment_attempts < 0:
            raise ValueError(
                "max_reassignment_attempts must be greater than or equal to 0."
            )

        if self.max_workers <= 0:
            raise ValueError(
                "max_workers must be greater than 0 when provided."
            )

        if self.director is None:
            self.director = self.setup_director()

    def agents_no_print(self):
        for agent in self.agents:
            agent.print_on = False

    def _context_for(self, agent_name: str) -> str:
        """What this agent has not been given yet. See context_utils."""
        return new_context_for(
            agent_name,
            self.conversation,
            self._delivered,
            empty_message="(no new messages)",
        )

    def _messages_for(self, agent_name: str) -> tuple:
        """Return typed prior messages and the latest task for an agent."""
        return split_last_turn(
            messages_for(agent_name, self.conversation),
            fallback="(no new messages)",
        )

    def _worker_run_payload(
        self, agent: Any, agent_name: str, task: str
    ) -> tuple:
        """Build a worker payload without duplicating conversation history."""
        if not self._agent_run_accepts(agent, "messages"):
            return task, {}
        return task, {
            "messages": messages_for(agent_name, self.conversation)
        }

    def _format_worker_responses(self, outputs: list) -> str:
        """Current-step worker results, not the full conversation log."""
        names = {
            self._agent_display_name(agent)
            for agent in (self.agents or [])
        }
        named = [
            f"{message.get('role')}: {message.get('content')}"
            for message in (
                self.conversation.conversation_history or []
            )
            if message.get("role") in names
        ]
        if named:
            if outputs:
                named = named[-len(outputs) :]
            return "\n\n".join(named)
        if not outputs:
            return "(no worker outputs)"
        return any_to_str(outputs)

    def _get_feedback_director(self) -> Agent:
        """Cached feedback director; built once per swarm instance."""
        if self._feedback_director is None:
            self._feedback_director = Agent(
                agent_name="Director",
                agent_description="Director module that provides feedback to the worker agents",
                model_name=self.feedback_director_model_name,
                max_loops=1,
                system_prompt=HIEARCHICAL_SWARM_SYSTEM_PROMPT,
                output_type="final",
            )
        return self._feedback_director

    def _get_judge_agent(self) -> Agent:
        """Cached judge agent; built once per swarm instance."""
        if self._judge_agent is None:
            self._judge_agent = Agent(
                agent_name="JudgeAgent",
                agent_description="Evaluates and scores the quality of worker agent outputs",
                system_prompt=HIERARCHICAL_SWARM_JUDGE_PROMPT,
                model_name=self.judge_agent_model_name,
                max_loops=1,
                base_model=JudgeReport,
                tools_list_dictionary=[_JUDGE_REPORT_SCHEMA],
                output_type="final",
            )
        return self._judge_agent

    def run_director(
        self,
        task: str,
        img: str = None,
    ) -> OrderBatch:
        """
        Run the director and record its orders.

        Args:
            task: Task to delegate.
            img: Optional image input.

        Returns:
            Director output containing worker orders.

        Raises:
            Exception: If director execution fails.
        """
        try:
            if self.planning_enabled is True:
                out = self.setup_director_with_planning(
                    task=self._director_task(task),
                    img=img,
                )
                self.conversation.add(
                    role=self.director.agent_name, content=out
                )

            # Run the director with the context
            function_call = self.director.run(
                task=self._director_task(task),
                img=img,
            )
            self._record_director_output(function_call)

            return function_call

        except Exception as e:
            logger.error(
                f"Hiearchical Swarm: Failed to run director: {e}"
            )
            raise

    async def arun_director(
        self,
        task: str,
        img: str = None,
    ) -> OrderBatch:
        """Async :meth:`run_director`."""
        try:
            if self.planning_enabled is True:
                out = await _arun_agent(
                    self._get_planning_director(),
                    task=self._director_task(task),
                    img=img,
                )
                self.conversation.add(
                    role=self.director.agent_name, content=out
                )

            function_call = await _arun_agent(
                self.director,
                task=self._director_task(task),
                img=img,
            )
            self._record_director_output(function_call)

            return function_call

        except Exception as e:
            logger.error(
                f"Hiearchical Swarm: Failed to run director: {e}"
            )
            raise

    def _director_task(self, task: str) -> str:
        """Wrap a task with the history the director has not seen yet."""
        return AGENT_TASK_TEMPLATE.format(
            history=self._context_for(self.director.agent_name),
            task=task,
        )

    def _record_director_output(self, function_call: Any) -> None:
        self.conversation.add(
            role="Director",
            content=(
                function_call
                if isinstance(function_call, str)
                else any_to_str(function_call)
            ),
        )

    def step(
        self,
        task: str,
        img: str = None,
        *args,
        is_final_loop: bool = False,
        **kwargs,
    ):
        """Run one director-worker-feedback cycle.

        Args:
            task: Task to process.
            img: Optional image input.
            *args: Worker positional arguments.
            is_final_loop: Skip feedback that cannot inform another loop.
            **kwargs: Worker keyword arguments.

        Returns:
            Worker outputs or director feedback.

        Raises:
            Exception: If step execution fails.
        """
        director_output = self.run_director(task=task, img=img)
        plan, orders = self.parse_orders(director_output)

        if self.print_on:
            formatter.print_director_task_distribution(
                director_name=self.director_name,
                orders=orders,
                plan=plan,
            )

        if not orders:
            return []

        outputs = self.execute_orders(orders)

        if self.agent_as_judge:
            return self.run_judge_agent(outputs)

        if (
            self.director_feedback_on
            and self.max_loops > 1
            and not is_final_loop
        ):
            return self.feedback_director(outputs)

        return outputs

    async def astep(
        self,
        task: str,
        img: str = None,
        *args,
        is_final_loop: bool = False,
        **kwargs,
    ):
        """Async :meth:`step`."""
        director_output = await self.arun_director(task=task, img=img)
        plan, orders = self.parse_orders(director_output)

        if self.print_on:
            formatter.print_director_task_distribution(
                director_name=self.director_name,
                orders=orders,
                plan=plan,
            )

        if not orders:
            return []

        outputs = await self.aexecute_orders(orders)

        if self.agent_as_judge:
            return await self.arun_judge_agent(outputs)

        if (
            self.director_feedback_on
            and self.max_loops > 1
            and not is_final_loop
        ):
            return await self.afeedback_director(outputs)

        return outputs

    @trace_run(
        "HierarchicalSwarm.run",
        input_params=("task", "tasks", "img", "imgs"),
    )
    def run(
        self,
        task: Optional[str] = None,
        img: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """Run the configured orchestration loops.

        Args:
            task: Initial task.
            img: Optional image input.
            *args: Worker positional arguments.
            **kwargs: Worker keyword arguments.

        Returns:
            Formatted conversation output.

        Raises:
            Exception: If swarm execution fails.
        """
        try:
            # Handle interactive mode task input
            if task is None and self.interactive:
                task = self._get_interactive_task()

            self._start_run(task)

            current_loop = 0
            last_output = None
            last_error = None
            any_loop_succeeded = False

            while current_loop < self.max_loops:
                try:
                    last_output = self.step(
                        task=self._loop_task(
                            current_loop, task, last_output
                        ),
                        img=img,
                        *args,
                        is_final_loop=(
                            current_loop == self.max_loops - 1
                        ),
                        **kwargs,
                    )
                    any_loop_succeeded = True

                except Exception as e:
                    last_error = e
                    last_output = None
                    logger.error(
                        f"[ERROR] Loop execution failed: {e} | Traceback: {traceback.format_exc()} | If this issue persists, please report it at: https://github.com/kyegomez/swarms/issues"
                    )

                current_loop += 1
                self._end_loop(current_loop, last_output, last_error)

            if not any_loop_succeeded and last_error is not None:
                raise last_error

            return self._finish_run()

        except Exception as e:

            self.workspace.save_conversation()
            logger.error(f"Hiearchical Swarm: Swarm run failed: {e}")
            raise

    async def arun(
        self,
        task: Optional[str] = None,
        img: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """Async :meth:`run`: agent calls are awaited, so the event loop stays free.

        Interactive mode reads the task through ``asyncio.to_thread``.
        """
        try:
            if task is None and self.interactive:
                task = await asyncio.to_thread(
                    self._get_interactive_task
                )

            self._start_run(task)

            current_loop = 0
            last_output = None
            last_error = None
            any_loop_succeeded = False

            while current_loop < self.max_loops:
                try:
                    last_output = await self.astep(
                        task=self._loop_task(
                            current_loop, task, last_output
                        ),
                        img=img,
                        *args,
                        is_final_loop=(
                            current_loop == self.max_loops - 1
                        ),
                        **kwargs,
                    )
                    any_loop_succeeded = True

                except Exception as e:
                    last_error = e
                    last_output = None
                    logger.error(
                        f"[ERROR] Loop execution failed: {e} | Traceback: {traceback.format_exc()} | If this issue persists, please report it at: https://github.com/kyegomez/swarms/issues"
                    )

                current_loop += 1
                self._end_loop(current_loop, last_output, last_error)

            if not any_loop_succeeded and last_error is not None:
                raise last_error

            return self._finish_run()

        except Exception as e:

            self.workspace.save_conversation()
            logger.error(f"Hiearchical Swarm: Swarm run failed: {e}")
            raise

    def _start_run(self, task: Optional[str]) -> None:
        """Reset per-run state and record the task."""
        self.conversation.clear()
        self._delivered = {}
        self.add_context_to_director()

        if task is not None:
            self.conversation.add(role="User", content=task)

    def _loop_task(
        self, current_loop: int, task: Optional[str], last_output: Any
    ) -> Optional[str]:
        if current_loop == 0:
            return task
        return LOOP_CONTINUATION_PROMPT.format(
            last_output=last_output, task=task
        )

    def _end_loop(
        self,
        current_loop: int,
        last_output: Any,
        last_error: Optional[Exception],
    ) -> None:
        """Add the loop completion marker to the conversation."""
        self.conversation.add(
            role="System",
            content=(
                f"--- Loop {current_loop}/{self.max_loops} failed: {last_error} ---"
                if last_output is None and last_error is not None
                else f"--- Loop {current_loop}/{self.max_loops} completed ---"
            ),
        )

    def _finish_run(self):
        result = history_output_formatter(
            conversation=self.conversation, type=self.output_type
        )
        self.workspace.save_conversation()
        return result

    def _get_interactive_task(self) -> str:
        """Read and return an interactive task."""
        print("\nEnter your task for the hierarchical swarm:")
        task = input("> ")
        return task.strip()

    def feedback_director(self, outputs: list):
        """
        Generate feedback for current worker outputs.

        Args:
            outputs: Worker outputs.

        Returns:
            Director feedback.

        Raises:
            Exception: If feedback generation fails.
        """
        try:
            output = self._get_feedback_director().run(
                task=DIRECTOR_FEEDBACK_PROMPT.format(
                    worker_responses=self._format_worker_responses(
                        outputs
                    )
                )
            )
            self.conversation.add(
                role=self.director.agent_name, content=output
            )

            return output

        except Exception as e:
            logger.error(
                f"Hiearchical Swarm: Feedback director failed: {e}"
            )

    async def afeedback_director(self, outputs: list):
        """Async :meth:`feedback_director`."""
        try:
            output = await _arun_agent(
                self._get_feedback_director(),
                task=DIRECTOR_FEEDBACK_PROMPT.format(
                    worker_responses=self._format_worker_responses(
                        outputs
                    )
                ),
            )
            self.conversation.add(
                role=self.director.agent_name, content=output
            )

            return output

        except Exception as e:
            logger.error(
                f"Hiearchical Swarm: Feedback director failed: {e}"
            )

    def run_judge_agent(self, outputs: list) -> str:
        """Score worker outputs with the cached judge.

        Args:
            outputs: Worker outputs used as an error fallback.

        Returns:
            Structured judge report.
        """
        try:
            logger.info(
                "Running judge agent to score worker outputs..."
            )
            judge = self._get_judge_agent()

            prior, judge_task = self._messages_for(judge.agent_name)
            result = judge.run(task=judge_task, messages=prior)
            self.conversation.add(role="JudgeAgent", content=result)
            logger.info(f"Judge agent completed scoring: {result}")
            return result

        except Exception as e:
            logger.error(
                f"[ERROR] run_judge_agent failed: {e} | Traceback: {traceback.format_exc()}"
            )
            return str(outputs)

    async def arun_judge_agent(self, outputs: list) -> str:
        """Async :meth:`run_judge_agent`."""
        try:
            logger.info(
                "Running judge agent to score worker outputs..."
            )
            judge = self._get_judge_agent()

            prior, judge_task = self._messages_for(judge.agent_name)
            result = await _arun_agent(
                judge, task=judge_task, messages=prior
            )
            self.conversation.add(role="JudgeAgent", content=result)
            logger.info(f"Judge agent completed scoring: {result}")
            return result

        except Exception as e:
            logger.error(
                f"[ERROR] arun_judge_agent failed: {e} | Traceback: {traceback.format_exc()}"
            )
            return str(outputs)

    def call_single_agent(
        self,
        agent_name: str,
        task: str,
        _add_to_conversation: bool = True,
        _raise_on_failure: bool = False,
        *args,
        **kwargs,
    ):
        """Run one worker by name.

        Args:
            agent_name: Worker name.
            task: Assigned task.
            *args: Worker positional arguments.
            **kwargs: Worker keyword arguments.

        Returns:
            Worker output.

        Raises:
            ValueError: If the worker is not found.
            Exception: If agent execution fails.
        """
        try:
            agent = self._find_worker(agent_name)
            worker_task, worker_extra = self._worker_run_payload(
                agent, agent_name, task
            )

            output = agent.run(
                *args,
                task=worker_task,
                **worker_extra,
                **kwargs,
            )
            if _add_to_conversation:
                self.conversation.add(role=agent_name, content=output)

            return output

        except Exception as e:

            logger.error(
                f"Hiearchical Swarm: Failed to call agent {agent_name}: {e}"
            )
            if _raise_on_failure:
                raise

    async def acall_single_agent(
        self,
        agent_name: str,
        task: str,
        _add_to_conversation: bool = True,
        _raise_on_failure: bool = False,
        *args,
        **kwargs,
    ):
        """Async :meth:`call_single_agent`."""
        try:
            agent = self._find_worker(agent_name)
            worker_task, worker_extra = self._worker_run_payload(
                agent, agent_name, task
            )

            output = await _arun_agent(
                agent,
                *args,
                task=worker_task,
                **worker_extra,
                **kwargs,
            )
            if _add_to_conversation:
                self.conversation.add(role=agent_name, content=output)

            return output

        except Exception as e:

            logger.error(
                f"Hiearchical Swarm: Failed to call agent {agent_name}: {e}"
            )
            if _raise_on_failure:
                raise

    def _record_agent_failure(
        self,
        order: HierarchicalOrder,
        error: Exception,
        attempts: int,
    ) -> Dict[str, Any]:
        """Record an unavailable worker in shared swarm context."""
        failure = {
            "status": "failed",
            "agent_name": order.agent_name,
            "task": order.task,
            "error": str(error),
            "attempts": attempts,
        }
        self.conversation.add(
            role="System",
            content=(
                "[WORKER UNAVAILABLE] "
                f"{order.agent_name} failed task {order.task!r} after "
                f"{attempts} attempt(s). Error: {error}. Do not assign new "
                "work to this agent during the current recovery cycle."
            ),
        )
        logger.warning(
            f"Worker {order.agent_name} is unavailable after "
            f"{attempts} attempt(s): {error}"
        )
        return failure

    def _execute_order_with_retries(
        self,
        order: HierarchicalOrder,
        add_to_conversation: bool = True,
    ):
        """Execute one order and return an explicit failure if retries expire."""
        attempts = self.max_agent_retries + 1
        last_error = None

        for attempt in range(1, attempts + 1):
            try:
                output = self.call_single_agent(
                    order.agent_name,
                    order.task,
                    _add_to_conversation=add_to_conversation,
                    _raise_on_failure=True,
                )
                return output, None
            except Exception as error:
                last_error = error
                if attempt < attempts:
                    logger.warning(
                        f"Retrying worker {order.agent_name} for task "
                        f"{order.task!r} ({attempt}/{attempts})"
                    )

        failure = self._record_agent_failure(
            order=order,
            error=last_error,
            attempts=attempts,
        )
        return failure, failure

    async def _aexecute_order_with_retries(
        self,
        order: HierarchicalOrder,
        add_to_conversation: bool = True,
    ):
        """Async :meth:`_execute_order_with_retries`."""
        attempts = self.max_agent_retries + 1
        last_error = None

        for attempt in range(1, attempts + 1):
            try:
                output = await self.acall_single_agent(
                    order.agent_name,
                    order.task,
                    _add_to_conversation=add_to_conversation,
                    _raise_on_failure=True,
                )
                return output, None
            except Exception as error:
                last_error = error
                if attempt < attempts:
                    logger.warning(
                        f"Retrying worker {order.agent_name} for task "
                        f"{order.task!r} ({attempt}/{attempts})"
                    )

        failure = self._record_agent_failure(
            order=order,
            error=last_error,
            attempts=attempts,
        )
        return failure, failure

    @staticmethod
    def _agent_display_name(agent: Any) -> str:
        """Leaf agents carry ``agent_name``; nested swarms carry ``name``."""
        return (
            getattr(agent, "agent_name", None)
            or getattr(agent, "name", None)
            or str(agent)
        )

    @staticmethod
    def _agent_run_accepts(agent: Any, parameter_name: str) -> bool:
        """Report whether ``agent.run`` accepts ``parameter_name``."""
        run_method = getattr(agent, "run", None)
        if run_method is None:
            return False
        try:
            signature = inspect.signature(run_method)
        except (TypeError, ValueError):
            # Uninspectable callables fail open rather than lose the kwarg.
            return True
        for parameter in signature.parameters.values():
            if parameter.name == parameter_name:
                return True
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                return True
        return False

    def _find_worker(self, agent_name: str) -> Any:
        """Find a worker by display name, leaf agent or nested swarm."""
        for agent in self.agents or []:
            if self._agent_display_name(agent) == agent_name:
                return agent
        raise ValueError(f"Agent with name '{agent_name}' not found")

    def _execute_orders_once(
        self,
        orders: List[HierarchicalOrder],
    ):
        """Execute a set of orders without triggering reassignment."""
        results = [None] * len(orders)
        failures = []

        if self.parallel_execution:
            futures_map = {}
            with ContextThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                for index, order in enumerate(orders):
                    future = executor.submit(
                        self._execute_order_with_retries,
                        order,
                        False,
                    )
                    futures_map[future] = (index, order)

                for future in as_completed(futures_map):
                    index, order = futures_map[future]
                    output, failure = future.result()
                    results[index] = output
                    if failure is not None:
                        failures.append(failure)

            self._add_parallel_results(orders, results)
        else:
            for index, order in enumerate(orders):
                output, failure = self._execute_order_with_retries(
                    order,
                )
                results[index] = output
                if failure is not None:
                    failures.append(failure)

        return results, failures

    async def _aexecute_orders_once(
        self,
        orders: List[HierarchicalOrder],
    ):
        """Async :meth:`_execute_orders_once`; at most ``max_workers`` orders run at once."""
        if not self.parallel_execution:
            pairs = [
                await self._aexecute_order_with_retries(order)
                for order in orders
            ]
        else:
            semaphore = asyncio.Semaphore(self.max_workers)

            async def run_order(order):
                async with semaphore:
                    return await self._aexecute_order_with_retries(
                        order, False
                    )

            pairs = await asyncio.gather(
                *(run_order(order) for order in orders)
            )
            self._add_parallel_results(
                orders, [output for output, _ in pairs]
            )

        results = [output for output, _ in pairs]
        failures = [
            failure for _, failure in pairs if failure is not None
        ]
        return results, failures

    def _add_parallel_results(
        self, orders: List[HierarchicalOrder], results: list
    ) -> None:
        """Add parallel outputs in order so history does not depend on timing."""
        for order, result in zip(orders, results):
            if result is not None and not (
                isinstance(result, dict)
                and result.get("status") == "failed"
            ):
                self.conversation.add(
                    role=order.agent_name,
                    content=result,
                )

    def _request_reassignment(
        self,
        failures: List[Dict[str, Any]],
        unavailable_agents: set,
    ) -> List[HierarchicalOrder]:
        """Ask the director to move failed work to available workers."""
        available_agents, recovery_task = self._reassignment_task(
            failures, unavailable_agents
        )
        if not available_agents:
            return []

        try:
            output = self.director.run(
                task=recovery_task,
                messages=messages_for(
                    getattr(
                        self.director,
                        "agent_name",
                        self.director_name,
                    ),
                    self.conversation,
                ),
            )
            self.conversation.add(role="Director", content=output)
            _, orders = self.parse_orders(output)
        except Exception as error:
            self._record_reassignment_failure(error)
            return []

        return self._valid_reassignment_orders(
            orders, available_agents
        )

    async def _arequest_reassignment(
        self,
        failures: List[Dict[str, Any]],
        unavailable_agents: set,
    ) -> List[HierarchicalOrder]:
        """Async :meth:`_request_reassignment`."""
        available_agents, recovery_task = self._reassignment_task(
            failures, unavailable_agents
        )
        if not available_agents:
            return []

        try:
            output = await _arun_agent(
                self.director,
                task=recovery_task,
                messages=messages_for(
                    getattr(
                        self.director,
                        "agent_name",
                        self.director_name,
                    ),
                    self.conversation,
                ),
            )
            self.conversation.add(role="Director", content=output)
            _, orders = self.parse_orders(output)
        except Exception as error:
            self._record_reassignment_failure(error)
            return []

        return self._valid_reassignment_orders(
            orders, available_agents
        )

    def _reassignment_task(
        self,
        failures: List[Dict[str, Any]],
        unavailable_agents: set,
    ) -> tuple:
        """Return the healthy worker names and the director's recovery prompt."""
        available_agents = [
            self._agent_display_name(agent)
            for agent in self.agents
            if self._agent_display_name(agent)
            not in unavailable_agents
        ]
        if not available_agents:
            self.conversation.add(
                role="System",
                content=(
                    "[RECOVERY STOPPED] No healthy worker agents remain "
                    "for reassignment."
                ),
            )
            return available_agents, None

        return available_agents, WORKER_RECOVERY_PROMPT.format(
            failures=json.dumps(failures, default=str),
            unavailable_agents=sorted(unavailable_agents),
            available_agents=available_agents,
        )

    def _record_reassignment_failure(self, error: Exception) -> None:
        self.conversation.add(
            role="System",
            content=(
                "[RECOVERY FAILED] The director could not produce "
                f"replacement orders: {error}"
            ),
        )
        logger.error(f"Director reassignment failed: {error}")

    def _valid_reassignment_orders(
        self,
        orders: List[HierarchicalOrder],
        available_agents: List[str],
    ) -> List[HierarchicalOrder]:
        """Drop replacement orders for unavailable or unknown workers."""
        valid_agent_names = set(available_agents)
        valid_orders = [
            order
            for order in orders
            if order.agent_name in valid_agent_names
        ]
        if len(valid_orders) != len(orders):
            self.conversation.add(
                role="System",
                content=(
                    "[RECOVERY NOTICE] Ignored replacement orders assigned "
                    "to unavailable or unknown agents."
                ),
            )
        return valid_orders

    def parse_orders(self, output):
        """Parse a director response into a plan and orders.

        Args:
            output: Raw director output.

        Returns:
            Plan and validated orders.

        Raises:
            ValueError: If parsing fails.
        """
        try:
            return _parse_orders(output)
        except Exception as e:
            logger.error(
                f"[ERROR] Failed to parse orders: {e} | Traceback: {traceback.format_exc()} | Report at: https://github.com/kyegomez/swarms/issues"
            )
            raise

    def execute_orders(
        self,
        orders: list,
    ):
        """Execute orders and recover failed assignments.

        Args:
            orders: Orders to execute.

        Returns:
            Order outputs.

        Raises:
            Exception: If order execution fails.
        """
        try:
            outputs, failures = self._execute_orders_once(
                orders=orders
            )
            unavailable_agents = {
                failure["agent_name"] for failure in failures
            }
            reassignment_attempt = 0

            while (
                failures
                and reassignment_attempt
                < self.max_reassignment_attempts
            ):
                reassignment_attempt += 1
                self._record_recovery_started(
                    failures, reassignment_attempt
                )
                replacement_orders = self._request_reassignment(
                    failures=failures,
                    unavailable_agents=unavailable_agents,
                )
                if not replacement_orders:
                    break

                replacement_outputs, failures = (
                    self._execute_orders_once(
                        orders=replacement_orders,
                    )
                )
                outputs.extend(replacement_outputs)
                unavailable_agents.update(
                    failure["agent_name"] for failure in failures
                )

            if failures:
                self._record_recovery_incomplete(failures)

            return outputs

        except Exception as e:
            return self._record_order_execution_error(e)

    async def aexecute_orders(
        self,
        orders: list,
    ):
        """Async :meth:`execute_orders`."""
        try:
            outputs, failures = await self._aexecute_orders_once(
                orders=orders
            )
            unavailable_agents = {
                failure["agent_name"] for failure in failures
            }
            reassignment_attempt = 0

            while (
                failures
                and reassignment_attempt
                < self.max_reassignment_attempts
            ):
                reassignment_attempt += 1
                self._record_recovery_started(
                    failures, reassignment_attempt
                )
                replacement_orders = (
                    await self._arequest_reassignment(
                        failures=failures,
                        unavailable_agents=unavailable_agents,
                    )
                )
                if not replacement_orders:
                    break

                replacement_outputs, failures = (
                    await self._aexecute_orders_once(
                        orders=replacement_orders,
                    )
                )
                outputs.extend(replacement_outputs)
                unavailable_agents.update(
                    failure["agent_name"] for failure in failures
                )

            if failures:
                self._record_recovery_incomplete(failures)

            return outputs

        except Exception as e:
            return self._record_order_execution_error(e)

    def _record_recovery_started(
        self, failures: list, reassignment_attempt: int
    ) -> None:
        self.conversation.add(
            role="System",
            content=(
                "[RECOVERY STARTED] Asking the director to reassign "
                f"{len(failures)} failed task(s). Recovery attempt "
                f"{reassignment_attempt}/"
                f"{self.max_reassignment_attempts}."
            ),
        )

    def _record_recovery_incomplete(self, failures: list) -> None:
        self.conversation.add(
            role="System",
            content=(
                "[RECOVERY INCOMPLETE] The swarm continued, but "
                f"{len(failures)} task(s) could not be completed."
            ),
        )

    def _record_order_execution_error(self, error: Exception) -> list:
        logger.error(
            f"[ERROR] Order execution failed: {error} | Traceback: {traceback.format_exc()} | If this issue persists, please report it at: https://github.com/kyegomez/swarms/issues"
        )
        self.conversation.add(
            role="System",
            content=(
                "[ORDER EXECUTION ERROR] The swarm continued after an "
                f"unexpected orchestration error: {error}"
            ),
        )
        return []

    def batched_run(
        self,
        tasks: List[str],
        *args,
        img: Optional[Union[str, List[Optional[str]]]] = None,
        imgs: Optional[List[Optional[str]]] = None,
        max_workers: Optional[int] = None,
        return_agent_output_dict: bool = False,
        return_exceptions: bool = False,
        **kwargs,
    ):
        """Run multiple tasks through the shared batch utility.

        Args:
            tasks: Tasks to execute.
            *args: Positional arguments forwarded to ``run``.
            img: One image for all tasks or one image per task.
            imgs: Images paired with tasks.
            max_workers: Concurrent task limit; ``None`` is sequential.
            return_agent_output_dict: Return results keyed by task.
            return_exceptions: Return exceptions instead of raising them.
            **kwargs: Keyword arguments forwarded to ``run``.

        Returns:
            Results in task order or keyed by task.

        Raises:
            ValueError: If batch utility arguments are invalid.
            Exception: If a task fails and exceptions are not returned.
        """
        return batched_run(
            self.run,
            tasks,
            *args,
            img=img,
            imgs=imgs,
            max_workers=max_workers,
            return_agent_output_dict=return_agent_output_dict,
            return_exceptions=return_exceptions,
            **kwargs,
        )
