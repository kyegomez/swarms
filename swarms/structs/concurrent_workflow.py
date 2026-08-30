import concurrent.futures
import time
from typing import Callable, List, Optional, Union

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.telemetry.otel import (
    ContextThreadPoolExecutor,
    capture_error,
    capture_init,
    trace_run,
)
from swarms.utils.formatter import formatter
from swarms.utils.generate_id import generate_id
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.workspace_manager import WorkspaceManager

logger = initialize_logger(log_folder="concurrent_workflow")

# Upper bound on concurrent agent calls when max_workers is not set. Agent
# calls are network-bound; this guards provider rate limits, not CPU.
MAX_CONCURRENT_AGENTS = 32


class ConcurrentWorkflow:
    """
    A concurrent workflow system for running multiple agents simultaneously.

    This class provides a framework for executing multiple agents concurrently on the same task,
    with optional dashboard monitoring, streaming callbacks, and various output formatting options.
    It uses ThreadPoolExecutor to manage concurrent execution and provides real-time status
    tracking for each agent.

    Attributes:
        id (str): Unique identifier for the workflow instance
        name (str): Human-readable name for the workflow
        description (str): Description of the workflow's purpose
        agents (List[Union[Agent, Callable]]): List of agents to execute concurrently
        auto_save (bool): Whether to automatically save workflow metadata
        output_type (str): Format for output formatting (e.g., "dict-all-except-first")
        max_loops (int): Maximum number of execution loops (currently unused)
        auto_generate_prompts (bool): Whether to enable automatic prompt engineering
        show_dashboard (bool): Whether to display real-time dashboard during execution
        agent_statuses (dict): Dictionary tracking status and output of each agent
        metadata_output_path (str): Path for saving workflow metadata
        autosave (bool): Whether to persist conversation history to the workflow
            workspace directory after each run
        verbose (bool): Whether to emit debug-level logging for workspace and
            autosave operations
        on_error (str): Failure policy for an agent that raises. ``"store"``
            records the error string as that agent's output and lets the other
            agents finish; ``"raise"`` propagates the exception and aborts the run
        max_workers (Optional[int]): Thread pool size for concurrent agent
            execution. Agent calls are network-bound, so this defaults to
            ``len(agents)`` capped at ``MAX_CONCURRENT_AGENTS`` rather than
            deriving from CPU count.
        swarm_workspace_dir (Optional[str]): Resolved workspace directory used for
            autosave, or None when autosave is disabled or setup failed
        conversation (Conversation): Conversation object for storing agent interactions

    Methods:
        run: Execute all agents concurrently on a given task
        batch_run: Execute workflow on multiple tasks sequentially
        run_with_dashboard: Execute agents with real-time dashboard monitoring
        cleanup: Clean up resources and connections
        fix_agents: Configure agents for dashboard mode
        reliability_check: Validate workflow configuration
        display_agent_dashboard: Display real-time dashboard

    Example:
        >>> from swarms import Agent, ConcurrentWorkflow
        >>>
        >>> # Create agents
        >>> agent1 = Agent(agent_name="Agent1", model_name="gpt-5.4")
        >>> agent2 = Agent(agent_name="Agent2", model_name="gpt-5.4")
        >>>
        >>> # Create workflow
        >>> workflow = ConcurrentWorkflow(
        ...     agents=[agent1, agent2],
        ...     show_dashboard=True
        ... )
        >>>
        >>> # Run workflow
        >>> result = workflow.run("Analyze this data")
    """

    def __init__(
        self,
        id: str = None,
        name: str = "ConcurrentWorkflow",
        description: str = "Execution of multiple agents concurrently",
        agents: List[Union[Agent, Callable]] = None,
        auto_save: bool = True,
        output_type: str = "dict-all-except-first",
        max_loops: int = 1,
        auto_generate_prompts: bool = False,
        show_dashboard: bool = False,
        autosave: bool = True,
        verbose: bool = False,
        on_error: str = "store",
        max_workers: Optional[int] = None,
    ):
        """
        Initialize a ConcurrentWorkflow.

        Args:
            id (str, optional): Unique identifier for this workflow instance.
                Generated automatically when omitted.
            name (str): Human-readable name for the workflow.
            description (str): Description of the workflow's purpose.
            agents (List[Union[Agent, Callable]], optional): Agents to execute
                concurrently on each task. Must be non-empty; validated by
                :meth:`reliability_check`.
            auto_save (bool): Whether to automatically save workflow metadata.
            output_type (str): Output formatting mode passed to
                ``history_output_formatter`` (e.g. ``"dict-all-except-first"``).
            max_loops (int): Maximum number of execution loops. Currently unused.
            auto_generate_prompts (bool): Whether to enable automatic prompt
                engineering for the configured agents.
            show_dashboard (bool): Whether to display a real-time dashboard during
                execution. Enabling this disables per-agent stdout printing to
                avoid conflicting with the dashboard.
            autosave (bool): Whether to persist conversation history to the
                workflow workspace directory after each run.
            verbose (bool): Whether to emit debug-level logging for workspace and
                autosave operations.
            on_error (str): Failure policy for an agent that raises. ``"store"``
                records the error string as that agent's output and lets the
                remaining agents finish; ``"raise"`` propagates the exception and
                aborts the run.
            max_workers (Optional[int]): Thread pool size for concurrent agent
                execution. Agent calls are network-bound rather than CPU-bound,
                so when omitted the pool is sized at ``len(agents)``, capped at
                ``MAX_CONCURRENT_AGENTS``. See :meth:`_resolve_max_workers`.

        Raises:
            ValueError: If no agents are provided or the agents list is empty.
        """
        self.id = id or generate_id("concurrent-workflow")
        self.name = name
        self.description = description
        self.agents = agents
        self.auto_save = auto_save
        self.max_loops = max_loops
        self.auto_generate_prompts = auto_generate_prompts
        self.output_type = output_type
        self.show_dashboard = show_dashboard
        self.autosave = autosave
        self.verbose = verbose
        self.on_error = on_error
        self.metadata_output_path = (
            f"concurrent_workflow_name_{name}_id_{self.id}.json"
        )
        self.max_workers = max_workers

        # Initialize agent statuses if agents are provided
        if agents is not None:
            self.agent_statuses = {
                agent.agent_name: {"status": "pending", "output": ""}
                for agent in agents
            }
        else:
            self.agent_statuses = {}

        self.reliability_check()
        self.conversation = Conversation(
            name=f"concurrent_workflow_name_{name}_id_{self.id}_conversation"
        )

        if self.show_dashboard is True:
            self.agents = self.fix_agents()

        self.workspace = WorkspaceManager(
            self,
            name=self.name or "concurrent-workflow",
            verbose=self.verbose,
            enabled=self.autosave,
        )
        self.swarm_workspace_dir = self.workspace.dir

        # Capture the full __init__ configuration if telemetry is enabled.
        capture_init(self)

    def fix_agents(self):
        """
        Configure agents for dashboard mode.

        Disables printing for all agents when dashboard mode is enabled to prevent
        console output conflicts with the dashboard display.

        Returns:
            List[Union[Agent, Callable]]: The configured list of agents.
        """
        if self.show_dashboard is True:
            for agent in self.agents:
                agent.print_on = False
        return self.agents

    def _resolve_max_workers(self) -> int:
        """
        Determine the thread pool size for concurrent agent execution.

        Each submitted task is one agent's LLM call, which is network-bound, so
        CPU count is not the limiting factor. The pool never receives more than
        ``len(self.agents)`` tasks, making that the natural size. The ceiling
        guards large rosters against provider rate limits and HTTP connection
        pool exhaustion.

        Returns:
            int: Number of worker threads to use, at least 1.
        """
        if self.max_workers is not None:
            return max(1, self.max_workers)
        return max(1, min(len(self.agents), MAX_CONCURRENT_AGENTS))

    def reliability_check(self):
        """
        Validate workflow configuration.

        Performs various validation checks to ensure the workflow is properly configured:
        - Checks that agents are provided
        - Validates that agents list is not empty
        - Warns if only one agent is provided (concurrent execution not beneficial)

        Raises:
            ValueError: If no agents are provided or agents list is empty.
            Exception: If any other validation error occurs.
        """
        try:
            if not self.agents or len(self.agents) == 0:
                raise ValueError(
                    "ConcurrentWorkflow: No agents provided"
                )
            if len(self.agents) == 1:
                logger.warning(
                    "ConcurrentWorkflow: Only one agent provided; concurrent execution may not be beneficial."
                )

        except Exception as e:
            logger.error(
                f"ConcurrentWorkflow: Reliability check failed: {e}"
            )
            raise

    def display_agent_dashboard(
        self,
        title: str = "ConcurrentWorkflow Dashboard",
        is_final: bool = False,
    ):
        """
        Display real-time dashboard showing agent status and outputs.

        Creates and displays a dashboard showing the current status and output
        of each agent in the workflow. This is used for monitoring concurrent execution.

        Args:
            title (str): Title to display for the dashboard. Defaults to "ConcurrentWorkflow Dashboard".
            is_final (bool): Whether this is the final dashboard display. Defaults to False.
        """
        agents_data = [
            {
                "name": agent.agent_name,
                "status": self.agent_statuses[agent.agent_name][
                    "status"
                ],
                "output": self.agent_statuses[agent.agent_name][
                    "output"
                ],
            }
            for agent in self.agents
        ]
        formatter.print_agent_dashboard(agents_data, title, is_final)

    def run_with_dashboard(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        streaming_callback: Optional[
            Callable[[str, str, bool], None]
        ] = None,
    ):
        """
        Execute agents with dashboard monitoring.

        Runs all agents concurrently while displaying a real-time dashboard that shows
        the status and output of each agent. This method provides visual feedback during
        execution and supports streaming callbacks for real-time updates.

        Args:
            task (str): The task to be executed by all agents.
            img (Optional[str]): Single image path for agents that support image input.
            imgs (Optional[List[str]]): List of image paths for agents that support multiple images.
            streaming_callback (Optional[Callable[[str, str, bool], None]]): Callback function for streaming updates.
                Called with (agent_name, chunk, is_final) parameters.

        Returns:
            Union[Dict, List, str]: Formatted conversation history based on output_type.
        """
        try:
            self.conversation.add(role="User", content=task)

            # Reset agent statuses
            for agent in self.agents:
                self.agent_statuses[agent.agent_name] = {
                    "status": "pending",
                    "output": "",
                }

            if self.show_dashboard:
                self.display_agent_dashboard()

            futures = []
            results = []

            def run_agent_with_status(agent, task, img, imgs):
                try:
                    self.agent_statuses[agent.agent_name][
                        "status"
                    ] = "running"
                    if self.show_dashboard:
                        self.display_agent_dashboard()

                    last_update_time = [0]
                    update_interval = 0.1

                    def agent_streaming_callback(chunk: str):
                        try:
                            if self.show_dashboard and chunk:
                                current_output = self.agent_statuses[
                                    agent.agent_name
                                ]["output"]
                                self.agent_statuses[agent.agent_name][
                                    "output"
                                ] = (current_output + chunk)

                                current_time = time.time()
                                if (
                                    current_time - last_update_time[0]
                                    >= update_interval
                                ):
                                    self.display_agent_dashboard()
                                    last_update_time[0] = current_time

                            if (
                                streaming_callback
                                and chunk is not None
                            ):
                                streaming_callback(
                                    agent.agent_name, chunk, False
                                )
                        except Exception as callback_error:
                            logger.warning(
                                f"Dashboard streaming callback failed for {agent.agent_name}: {str(callback_error)}"
                            )

                    output = agent.run(
                        task=task,
                        img=img,
                        imgs=imgs,
                        streaming_callback=agent_streaming_callback,
                    )

                    self.agent_statuses[agent.agent_name][
                        "status"
                    ] = "completed"
                    self.agent_statuses[agent.agent_name][
                        "output"
                    ] = output
                    if self.show_dashboard:
                        self.display_agent_dashboard()

                    if streaming_callback:
                        streaming_callback(agent.agent_name, "", True)

                    return output
                except Exception as e:
                    self.agent_statuses[agent.agent_name][
                        "status"
                    ] = "error"
                    self.agent_statuses[agent.agent_name][
                        "output"
                    ] = f"Error: {str(e)}"
                    if self.show_dashboard:
                        self.display_agent_dashboard()

                    if streaming_callback:
                        streaming_callback(
                            agent.agent_name, f"Error: {str(e)}", True
                        )

                    raise

            with ContextThreadPoolExecutor(
                max_workers=self._resolve_max_workers()
            ) as executor:
                futures = [
                    executor.submit(
                        run_agent_with_status, agent, task, img, imgs
                    )
                    for agent in self.agents
                ]
                concurrent.futures.wait(futures)

                for future, agent in zip(futures, self.agents):
                    try:
                        output = future.result()
                        results.append((agent.agent_name, output))
                    except Exception as e:
                        # Same failure policy as _run: the dashboard must not revoke on_error.
                        if self.on_error == "raise":
                            raise
                        capture_error(
                            e,
                            self,
                            name="ConcurrentWorkflow.agent_error",
                            agent=getattr(agent, "agent_name", None),
                        )
                        logger.error(
                            f"Agent {agent.agent_name} failed: {str(e)}"
                        )
                        results.append(
                            (agent.agent_name, f"Error: {str(e)}")
                        )

            for agent_name, output in results:
                self.conversation.add(role=agent_name, content=output)

            if self.show_dashboard:
                self.display_agent_dashboard(
                    "Final ConcurrentWorkflow Dashboard",
                    is_final=True,
                )

            return history_output_formatter(
                conversation=self.conversation, type=self.output_type
            )
        finally:
            if self.show_dashboard:
                formatter.stop_dashboard()

    def _run(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        streaming_callback: Optional[
            Callable[[str, str, bool], None]
        ] = None,
    ):
        """
        Execute agents concurrently without dashboard.

        Internal method that runs all agents concurrently using ThreadPoolExecutor
        without displaying the dashboard. This is the core execution logic used when
        dashboard mode is disabled.

        Args:
            task (str): The task to be executed by all agents.
            img (Optional[str]): Single image path for agents that support image input.
            imgs (Optional[List[str]]): List of image paths for agents that support multiple images.
            streaming_callback (Optional[Callable[[str, str, bool], None]]): Callback function for streaming updates.

        Returns:
            Union[Dict, List, str]: Formatted conversation history based on output_type.
        """
        self.conversation.add(role="User", content=task)

        with ContextThreadPoolExecutor(
            max_workers=self._resolve_max_workers()
        ) as executor:
            future_to_agent = {
                executor.submit(
                    self._run_agent_with_streaming,
                    agent,
                    task,
                    img,
                    imgs,
                    streaming_callback,
                ): agent
                for agent in self.agents
            }

            for future in concurrent.futures.as_completed(
                future_to_agent
            ):
                agent = future_to_agent[future]
                try:
                    output = future.result()
                    self.conversation.add(
                        role=agent.agent_name, content=output
                    )
                except Exception as e:
                    if self.on_error == "raise":
                        raise
                    # Track the swallowed per-agent failure so it isn't lost.
                    capture_error(
                        e,
                        self,
                        name="ConcurrentWorkflow.agent_error",
                        agent=getattr(agent, "agent_name", None),
                    )
                    logger.error(
                        f"Agent {agent.agent_name} failed: {str(e)}"
                    )
                    self.conversation.add(
                        role=f"{agent.agent_name} (failed)",
                        content=f"Error: {str(e)}",
                    )

        return history_output_formatter(
            conversation=self.conversation, type=self.output_type
        )

    def _run_agent_with_streaming(
        self,
        agent: Union[Agent, Callable],
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        streaming_callback: Optional[
            Callable[[str, str, bool], None]
        ] = None,
    ):
        """
        Run single agent with streaming support.

        Executes a single agent with optional streaming callback support.
        Handles errors gracefully and ensures completion callbacks are called.

        Args:
            agent (Union[Agent, Callable]): The agent to execute.
            task (str): The task to be executed by the agent.
            img (Optional[str]): Single image path for agents that support image input.
            imgs (Optional[List[str]]): List of image paths for agents that support multiple images.
            streaming_callback (Optional[Callable[[str, str, bool], None]]): Callback function for streaming updates.

        Returns:
            str: The output from the agent.

        Raises:
            Exception: If the agent execution fails.
        """
        if streaming_callback is None:
            return agent.run(task=task, img=img, imgs=imgs)

        def agent_streaming_callback(chunk: str):
            try:
                # Safely call the streaming callback
                if streaming_callback and chunk is not None:
                    streaming_callback(agent.agent_name, chunk, False)
            except Exception as callback_error:
                logger.warning(
                    f"Streaming callback failed for {agent.agent_name}: {str(callback_error)}"
                )

        try:
            output = agent.run(
                task=task,
                img=img,
                imgs=imgs,
                streaming_callback=agent_streaming_callback,
            )
            # Ensure completion callback is called even if there were issues
            try:
                streaming_callback(agent.agent_name, "", True)
            except Exception as callback_error:
                logger.warning(
                    f"Completion callback failed for {agent.agent_name}: {str(callback_error)}"
                )
            return output
        except Exception as e:
            error_msg = f"Agent {agent.agent_name} failed: {str(e)}"
            logger.error(error_msg)
            # Try to send error through callback
            try:
                streaming_callback(
                    agent.agent_name, f"Error: {str(e)}", True
                )
            except Exception as callback_error:
                logger.warning(
                    f"Error callback failed for {agent.agent_name}: {str(callback_error)}"
                )
            raise

    @trace_run(
        "ConcurrentWorkflow.run", input_params=("task", "img", "imgs")
    )
    def run(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        streaming_callback: Optional[
            Callable[[str, str, bool], None]
        ] = None,
    ):
        """
        Execute all agents concurrently.

        Main entry point for running the concurrent workflow. Executes all agents
        simultaneously on the given task, with optional dashboard monitoring and
        streaming callbacks. Automatically cleans up resources after execution.

        Args:
            task (str): The task to be executed by all agents.
            img (Optional[str]): Single image path for agents that support image input.
            imgs (Optional[List[str]]): List of image paths for agents that support multiple images.
            streaming_callback (Optional[Callable[[str, str, bool], None]]): Callback function for streaming updates.
                Called with (agent_name, chunk, is_final) parameters.

        Returns:
            Union[Dict, List, str]: Formatted conversation history based on output_type.

        Example:
            >>> workflow = ConcurrentWorkflow(agents=[agent1, agent2])
            >>> result = workflow.run("Analyze this data")
        """
        try:
            if self.show_dashboard:
                result = self.run_with_dashboard(
                    task, img, imgs, streaming_callback
                )
            else:
                result = self._run(
                    task, img, imgs, streaming_callback
                )

            self.workspace.save_conversation()

            return result
        except Exception:
            self.workspace.save_conversation()
            raise

    def batch_run(
        self,
        tasks: List[str],
        imgs: Optional[List[str]] = None,
        streaming_callback: Optional[
            Callable[[str, str, bool], None]
        ] = None,
    ):
        """
        Execute workflow on multiple tasks sequentially.

        Runs the concurrent workflow on multiple tasks one after another.
        Each task is executed with all agents running concurrently, but the tasks
        themselves are processed sequentially.

        Args:
            tasks (List[str]): List of tasks to be executed.
            imgs (Optional[List[str]]): List of image paths corresponding to each task.
            streaming_callback (Optional[Callable[[str, str, bool], None]]): Callback function for streaming updates.

        Returns:
            List[Union[Dict, List, str]]: List of results for each task.

        Example:
            >>> workflow = ConcurrentWorkflow(agents=[agent1, agent2])
            >>> results = workflow.batch_run(["Task 1", "Task 2", "Task 3"])
        """
        results = []
        for idx, task in enumerate(tasks):
            img = None
            if imgs is not None and idx < len(imgs):
                img = imgs[idx]
            results.append(
                self.run(
                    task=task,
                    img=img,
                    streaming_callback=streaming_callback,
                )
            )
        return results
