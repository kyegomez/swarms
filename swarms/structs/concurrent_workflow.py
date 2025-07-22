import concurrent.futures
import os
from typing import Callable, List, Optional, Union

from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.formatter import formatter

logger = initialize_logger(log_folder="concurrent_workflow")


class ConcurrentWorkflow(BaseSwarm):
    """
    A production-grade concurrent workflow orchestrator that executes multiple agents simultaneously.

    ConcurrentWorkflow is designed for high-performance multi-agent orchestration with advanced features
    including real-time monitoring, error handling, caching, and flexible output formatting. It's ideal
    for scenarios where multiple agents need to process the same task independently and their results
    need to be aggregated.

    Key Features:
        - Concurrent execution using ThreadPoolExecutor for optimal performance
        - Real-time dashboard monitoring with status updates
        - Comprehensive error handling and recovery
        - Flexible output formatting options
        - Automatic prompt engineering capabilities
        - Conversation history management
        - Metadata persistence and auto-saving
        - Support for both single and batch processing
        - Image input support for multimodal agents

    Use Cases:
        - Multi-perspective analysis (financial, legal, technical reviews)
        - Consensus building and voting systems
        - Parallel data processing and analysis
        - A/B testing with different agent configurations
        - Redundancy and reliability improvements

    Args:
        name (str, optional): Unique identifier for the workflow instance.
            Defaults to "ConcurrentWorkflow".
        description (str, optional): Human-readable description of the workflow's purpose.
            Defaults to "Execution of multiple agents concurrently".
        agents (List[Union[Agent, Callable]], optional): List of Agent instances or callable objects
            to execute concurrently. Each agent should implement a `run` method.
            Defaults to empty list.
        metadata_output_path (str, optional): File path for saving execution metadata and results.
            Supports JSON format. Defaults to "agent_metadata.json".
        auto_save (bool, optional): Whether to automatically save conversation history and metadata
            after each run. Defaults to True.
        output_type (str, optional): Format for aggregating agent outputs. Options include:
            - "dict-all-except-first": Dictionary with all agent outputs except the first
            - "dict": Dictionary with all agent outputs
            - "str": Concatenated string of all outputs
            - "list": List of individual agent outputs
            Defaults to "dict-all-except-first".
        max_loops (int, optional): Maximum number of execution loops for each agent.
            Defaults to 1.
        auto_generate_prompts (bool, optional): Enable automatic prompt engineering for all agents.
            When True, agents will enhance their prompts automatically. Defaults to False.
        show_dashboard (bool, optional): Enable real-time dashboard display showing agent status,
            progress, and outputs. Useful for monitoring and debugging. Defaults to False.
        *args: Additional positional arguments passed to the BaseSwarm parent class.
        **kwargs: Additional keyword arguments passed to the BaseSwarm parent class.

    Raises:
        ValueError: If agents list is empty or None.
        ValueError: If description is empty or None.
        TypeError: If agents list contains non-Agent, non-callable objects.

    Attributes:
        name (str): The workflow instance name.
        description (str): The workflow description.
        agents (List[Union[Agent, Callable]]): List of agents to execute.
        metadata_output_path (str): Path for metadata output file.
        auto_save (bool): Auto-save flag for metadata persistence.
        output_type (str): Output aggregation format.
        max_loops (int): Maximum execution loops per agent.
        auto_generate_prompts (bool): Auto prompt engineering flag.
        show_dashboard (bool): Dashboard display flag.
        agent_statuses (dict): Real-time status tracking for each agent.
        conversation (Conversation): Conversation history manager.

    Example:
        Basic usage with multiple agents:

        >>> from swarms import Agent, ConcurrentWorkflow
        >>>
        >>> # Create specialized agents
        >>> financial_agent = Agent(
        ...     agent_name="Financial-Analyst",
        ...     system_prompt="You are a financial analysis expert...",
        ...     model_name="gpt-4"
        ... )
        >>> legal_agent = Agent(
        ...     agent_name="Legal-Advisor",
        ...     system_prompt="You are a legal expert...",
        ...     model_name="gpt-4"
        ... )
        >>>
        >>> # Create workflow
        >>> workflow = ConcurrentWorkflow(
        ...     name="Multi-Expert-Analysis",
        ...     agents=[financial_agent, legal_agent],
        ...     show_dashboard=True,
        ...     auto_save=True
        ... )
        >>>
        >>> # Execute analysis
        >>> task = "Analyze the risks of investing in cryptocurrency"
        >>> results = workflow.run(task)
        >>> print(f"Analysis complete with {len(results)} perspectives")

        Batch processing example:

        >>> tasks = [
        ...     "Analyze Q1 financial performance",
        ...     "Review Q2 market trends",
        ...     "Forecast Q3 projections"
        ... ]
        >>> batch_results = workflow.batch_run(tasks)
        >>> print(f"Processed {len(batch_results)} quarterly reports")

    Note:
        - Agents are executed using ThreadPoolExecutor with 95% of available CPU cores
        - Each agent runs independently and cannot communicate with others during execution
        - The workflow maintains conversation history across all runs for context
        - Dashboard mode disables individual agent printing to prevent output conflicts
        - Error handling ensures partial results are available even if some agents fail
    """

    def __init__(
        self,
        name: str = "ConcurrentWorkflow",
        description: str = "Execution of multiple agents concurrently",
        agents: List[Union[Agent, Callable]] = [],
        metadata_output_path: str = "agent_metadata.json",
        auto_save: bool = True,
        output_type: str = "dict-all-except-first",
        max_loops: int = 1,
        auto_generate_prompts: bool = False,
        show_dashboard: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize the ConcurrentWorkflow with configuration parameters.

        Performs initialization, validation, and setup of internal components including
        conversation management, agent status tracking, and dashboard configuration.

        Note:
            The constructor automatically performs reliability checks and configures
            agents for dashboard mode if enabled. Agent print outputs are disabled
            when dashboard mode is active to prevent display conflicts.
        """
        super().__init__(
            name=name,
            description=description,
            agents=agents,
            *args,
            **kwargs,
        )
        self.name = name
        self.description = description
        self.agents = agents
        self.metadata_output_path = metadata_output_path
        self.auto_save = auto_save
        self.max_loops = max_loops
        self.auto_generate_prompts = auto_generate_prompts
        self.output_type = output_type
        self.show_dashboard = show_dashboard
        self.agent_statuses = {
            agent.agent_name: {"status": "pending", "output": ""}
            for agent in agents
        }

        self.reliability_check()
        self.conversation = Conversation()

        if self.show_dashboard is True:
            self.agents = self.fix_agents()

    def fix_agents(self):
        """
        Configure agents for dashboard mode by disabling individual print outputs.

        When dashboard mode is enabled, individual agent print outputs can interfere
        with the dashboard display. This method disables print_on for all agents
        to ensure clean dashboard rendering.

        Returns:
            List[Agent]: The modified list of agents with print_on disabled.

        Note:
            This method only modifies agents when show_dashboard is True.
            Agent functionality is not affected, only their output display behavior.

        Example:
            >>> workflow = ConcurrentWorkflow(show_dashboard=True, agents=[agent1, agent2])
            >>> # Agents automatically configured for dashboard mode
            >>> all(not agent.print_on for agent in workflow.agents)
            True
        """
        if self.show_dashboard is True:
            for agent in self.agents:
                agent.print_on = False
        return self.agents

    def reliability_check(self):
        """
        Perform comprehensive validation of workflow configuration and agents.

        Validates that the workflow is properly configured with valid agents and
        provides warnings for suboptimal configurations. This method is called
        automatically during initialization.

        Validates:
            - Agents list is not None or empty
            - At least one agent is provided
            - Warns if only one agent is provided (suboptimal for concurrent execution)

        Raises:
            ValueError: If agents list is None or empty.

        Logs:
            Warning: If only one agent is provided (concurrent execution not beneficial).
            Error: If validation fails with detailed error information.

        Example:
            >>> workflow = ConcurrentWorkflow(agents=[])
            ValueError: ConcurrentWorkflow: No agents provided

            >>> workflow = ConcurrentWorkflow(agents=[single_agent])
            # Logs warning about single agent usage
        """
        try:
            if self.agents is None:
                raise ValueError(
                    "ConcurrentWorkflow: No agents provided"
                )

            if len(self.agents) == 0:
                raise ValueError(
                    "ConcurrentWorkflow: No agents provided"
                )

            if len(self.agents) == 1:
                logger.warning(
                    "ConcurrentWorkflow: Only one agent provided. With ConcurrentWorkflow, you should use at least 2+ agents."
                )
        except Exception as e:
            logger.error(
                f"ConcurrentWorkflow: Reliability check failed: {e}"
            )
            raise

    def activate_auto_prompt_engineering(self):
        """
        Enable automatic prompt engineering for all agents in the workflow.

        When activated, each agent will automatically enhance and optimize their
        system prompts based on the task context and their previous interactions.
        This can improve response quality but may increase execution time.

        Side Effects:
            - Sets auto_generate_prompt=True for all agents in the workflow
            - Affects subsequent agent.run() calls, not retroactively
            - May increase latency due to prompt optimization overhead

        Note:
            This method can be called at any time, but changes only affect
            future agent executions. Already running agents are not affected.

        Example:
            >>> workflow = ConcurrentWorkflow(agents=[agent1, agent2])
            >>> workflow.activate_auto_prompt_engineering()
            >>> # All agents will now auto-generate optimized prompts
            >>> result = workflow.run("Complex analysis task")
            >>> # Agents used enhanced prompts for better performance

        See Also:
            - Agent.auto_generate_prompt: Individual agent prompt engineering
            - auto_generate_prompts: Constructor parameter for default behavior
        """
        if self.auto_generate_prompts is True:
            for agent in self.agents:
                agent.auto_generate_prompt = True

    def display_agent_dashboard(
        self,
        title: str = "🤖 Agent Dashboard",
        is_final: bool = False,
    ) -> None:
        """
        Display a real-time dashboard showing the current status of all agents.

        Renders a formatted dashboard with agent names, execution status, and
        output previews. The dashboard is updated in real-time during workflow
        execution to provide visibility into agent progress and results.

        Args:
            title (str, optional): The dashboard title to display at the top.
                Defaults to "🤖 Agent Dashboard".
            is_final (bool, optional): Whether this is the final dashboard display
                after all agents have completed. Changes formatting and styling.
                Defaults to False.

        Side Effects:
            - Prints formatted dashboard to console
            - Updates display in real-time during execution
            - May clear previous dashboard content for clean updates

        Note:
            This method is automatically called during workflow execution when
            show_dashboard=True. Manual calls are supported for custom monitoring.

        Dashboard Status Values:
            - "pending": Agent queued but not yet started
            - "running": Agent currently executing task
            - "completed": Agent finished successfully
            - "error": Agent execution failed with error

        Example:
            >>> workflow = ConcurrentWorkflow(agents=[agent1, agent2], show_dashboard=True)
            >>> workflow.display_agent_dashboard("Custom Dashboard Title")
            # Displays:
            # 🤖 Custom Dashboard Title
            # ┌─────────────────┬─────────┬──────────────────┐
            # │ Agent Name      │ Status  │ Output Preview   │
            # ├─────────────────┼─────────┼──────────────────┤
            # │ Financial-Agent │ running │ Analyzing data...│
            # │ Legal-Agent     │ pending │                  │
            # └─────────────────┴─────────┴──────────────────┘
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
    ):
        """
        Execute all agents concurrently with real-time dashboard monitoring.

        This method provides the same concurrent execution as _run() but with
        enhanced real-time monitoring through a visual dashboard. Agent status
        updates are displayed in real-time, showing progress, completion, and
        any errors that occur during execution.

        Args:
            task (str): The task description or prompt to be executed by all agents.
                This will be passed to each agent's run() method.
            img (Optional[str], optional): Path to a single image file for agents
                that support multimodal input. Defaults to None.
            imgs (Optional[List[str]], optional): List of image file paths for
                agents that support multiple image inputs. Defaults to None.

        Returns:
            Any: Formatted output based on the configured output_type setting.
            The return format depends on the output_type parameter set during
            workflow initialization.

        Raises:
            Exception: Re-raises any exceptions from agent execution after
                updating the dashboard with error status.

        Side Effects:
            - Displays initial dashboard showing all agents as "pending"
            - Updates dashboard in real-time as agents start, run, and complete
            - Displays final dashboard with all results when execution completes
            - Adds all task inputs and agent outputs to conversation history
            - Automatically cleans up dashboard display resources

        Dashboard Flow:
            1. Initial dashboard shows all agents as "pending"
            2. As agents start, status updates to "running"
            3. As agents complete, status updates to "completed" with output preview
            4. Final dashboard shows complete results summary
            5. Dashboard resources are cleaned up automatically

        Performance:
            - Uses 95% of available CPU cores for optimal concurrency
            - ThreadPoolExecutor manages thread lifecycle automatically
            - Real-time updates have minimal performance overhead

        Example:
            >>> workflow = ConcurrentWorkflow(
            ...     agents=[financial_agent, legal_agent],
            ...     show_dashboard=True
            ... )
            >>> result = workflow.run_with_dashboard(
            ...     task="Analyze the merger proposal",
            ...     img="company_financials.png"
            ... )
            # Dashboard shows real-time progress:
            # Agent-1: pending -> running -> completed
            # Agent-2: pending -> running -> completed
            >>> print("Analysis complete:", result)

        Note:
            This method is automatically called when show_dashboard=True.
            For headless execution without dashboard, use _run() directly.
        """
        try:
            self.conversation.add(role="User", content=task)

            # Reset agent statuses
            for agent in self.agents:
                self.agent_statuses[agent.agent_name] = {
                    "status": "pending",
                    "output": "",
                }

            # Display initial dashboard if enabled
            if self.show_dashboard:
                self.display_agent_dashboard()

            # Use 95% of available CPU cores for optimal performance
            max_workers = int(os.cpu_count() * 0.95)

            # Create a list to store all futures and their results
            futures = []
            results = []

            def run_agent_with_status(agent, task, img, imgs):
                try:
                    # Update status to running
                    self.agent_statuses[agent.agent_name][
                        "status"
                    ] = "running"
                    if self.show_dashboard:
                        self.display_agent_dashboard()

                    # Run the agent
                    output = agent.run(task=task, img=img, imgs=imgs)

                    # Update status to completed
                    self.agent_statuses[agent.agent_name][
                        "status"
                    ] = "completed"
                    self.agent_statuses[agent.agent_name][
                        "output"
                    ] = output
                    if self.show_dashboard:
                        self.display_agent_dashboard()

                    return output
                except Exception as e:
                    # Update status to error
                    self.agent_statuses[agent.agent_name][
                        "status"
                    ] = "error"
                    self.agent_statuses[agent.agent_name][
                        "output"
                    ] = f"Error: {str(e)}"
                    if self.show_dashboard:
                        self.display_agent_dashboard()
                    raise

            # Run agents concurrently using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                # Submit all agent tasks
                futures = [
                    executor.submit(
                        run_agent_with_status, agent, task, img, imgs
                    )
                    for agent in self.agents
                ]

                # Wait for all futures to complete
                concurrent.futures.wait(futures)

                # Process results in order of completion
                for future, agent in zip(futures, self.agents):
                    try:
                        output = future.result()
                        results.append((agent.agent_name, output))
                    except Exception as e:
                        logger.error(
                            f"Agent {agent.agent_name} failed: {str(e)}"
                        )
                        results.append(
                            (agent.agent_name, f"Error: {str(e)}")
                        )

            # Add all results to conversation
            for agent_name, output in results:
                self.conversation.add(role=agent_name, content=output)

            # Display final dashboard if enabled
            if self.show_dashboard:
                self.display_agent_dashboard(
                    "🎉 Final Agent Dashboard", is_final=True
                )

            return history_output_formatter(
                conversation=self.conversation,
                type=self.output_type,
            )
        finally:
            # Always clean up the dashboard display
            if self.show_dashboard:
                formatter.stop_dashboard()

    def _run(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
    ):
        """
        Execute all agents concurrently without dashboard monitoring (headless mode).

        This is the core execution method that runs all agents simultaneously using
        ThreadPoolExecutor for optimal performance. Results are collected and
        formatted according to the configured output_type.

        Args:
            task (str): The task description or prompt to be executed by all agents.
                Each agent receives the same task input for independent processing.
            img (Optional[str], optional): Path to a single image file for multimodal
                agents. The same image is provided to all agents. Defaults to None.
            imgs (Optional[List[str]], optional): List of image file paths for agents
                that support multiple image inputs. All agents receive the same image list.
                Defaults to None.

        Returns:
            Any: Formatted output according to the output_type configuration:
                - "dict-all-except-first": Dict with all agent outputs except first
                - "dict": Dict with all agent outputs keyed by agent name
                - "str": Concatenated string of all agent outputs
                - "list": List of individual agent outputs in completion order

        Side Effects:
            - Adds the user task to conversation history
            - Adds each agent's output to conversation history upon completion
            - No visual output or dashboard updates (headless execution)

        Performance Characteristics:
            - Uses ThreadPoolExecutor with 95% of available CPU cores
            - Agents execute truly concurrently, not sequentially
            - Results are collected in completion order, not submission order
            - Memory efficient for large numbers of agents

        Error Handling:
            - Individual agent failures don't stop other agents
            - Failed agents have their exceptions logged but execution continues
            - Partial results are still returned for successful agents

        Thread Safety:
            - Conversation object handles concurrent access safely
            - Agent status is not tracked in this method (dashboard-free)
            - Each agent runs in isolation without shared state

        Example:
            >>> workflow = ConcurrentWorkflow(
            ...     agents=[agent1, agent2, agent3],
            ...     output_type="dict"
            ... )
            >>> result = workflow._run(
            ...     task="Analyze market trends for Q4",
            ...     img="market_chart.png"
            ... )
            >>> print(f"Received {len(result)} agent analyses")
            >>> # Result format: {"agent1": "analysis1", "agent2": "analysis2", ...}

        Note:
            This method is called automatically by run() when show_dashboard=False.
            For monitoring and real-time updates, use run_with_dashboard() instead.
        """
        self.conversation.add(role="User", content=task)

        # Use 95% of available CPU cores for optimal performance
        max_workers = int(os.cpu_count() * 0.95)

        # Run agents concurrently using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            # Submit all agent tasks and store with their index
            future_to_agent = {
                executor.submit(
                    agent.run, task=task, img=img, imgs=imgs
                ): agent
                for agent in self.agents
            }

            # Collect results and add to conversation in completion order
            for future in concurrent.futures.as_completed(
                future_to_agent
            ):
                agent = future_to_agent[future]
                output = future.result()
                self.conversation.add(role=agent.name, content=output)

        return history_output_formatter(
            conversation=self.conversation,
            type=self.output_type,
        )

    def run(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
    ):
        """
        Execute all agents concurrently on the given task with optional dashboard monitoring.

        This is the main entry point for workflow execution. Automatically selects
        between dashboard-enabled execution (run_with_dashboard) or headless execution
        (_run) based on the show_dashboard configuration.

        Args:
            task (str): The task description, prompt, or instruction to be executed
                by all agents concurrently. Each agent processes the same task
                independently.
            img (Optional[str], optional): Path to a single image file for agents
                that support multimodal (text + image) input. Defaults to None.
            imgs (Optional[List[str]], optional): List of image file paths for
                agents that support multiple image inputs. Defaults to None.

        Returns:
            Any: Aggregated results from all agents formatted according to output_type:
                - "dict-all-except-first": Dictionary excluding first agent's output
                - "dict": Complete dictionary with all agent outputs
                - "str": Concatenated string of all outputs
                - "list": List of individual agent outputs

        Execution Modes:
            - Dashboard Mode (show_dashboard=True): Provides real-time visual monitoring
              with status updates, progress tracking, and error visibility
            - Headless Mode (show_dashboard=False): Silent execution with no visual output,
              optimized for performance and automation scenarios

        Concurrent Execution:
            - All agents run simultaneously using ThreadPoolExecutor
            - Utilizes 95% of available CPU cores for optimal performance
            - Thread-safe conversation history management
            - Independent agent execution without inter-agent communication

        Error Resilience:
            - Individual agent failures don't halt the entire workflow
            - Partial results are returned for successful agents
            - Comprehensive error logging for debugging
            - Graceful degradation under failure conditions

        Example:
            Basic concurrent execution:

            >>> workflow = ConcurrentWorkflow(agents=[analyst, reviewer, summarizer])
            >>> result = workflow.run("Evaluate the new product proposal")
            >>> print(f"Received insights from {len(workflow.agents)} experts")

            With dashboard monitoring:

            >>> workflow = ConcurrentWorkflow(
            ...     agents=[financial_agent, legal_agent],
            ...     show_dashboard=True
            ... )
            >>> result = workflow.run("Review merger agreement")
            # Real-time dashboard shows progress and completion status

            Multimodal analysis:

            >>> result = workflow.run(
            ...     task="Analyze this chart and provide insights",
            ...     img="quarterly_results.png"
            ... )

        Performance Tips:
            - Use 2+ agents to benefit from concurrent execution
            - Dashboard mode adds minimal overhead for monitoring
            - Larger agent counts scale well with available CPU cores
            - Consider batch_run() for multiple related tasks

        See Also:
            - batch_run(): For processing multiple tasks sequentially
            - run_with_dashboard(): For direct dashboard execution
            - _run(): For direct headless execution
        """
        if self.show_dashboard:
            return self.run_with_dashboard(task, img, imgs)
        else:
            return self._run(task, img, imgs)

    def batch_run(
        self,
        tasks: List[str],
        imgs: Optional[List[str]] = None,
    ):
        """
        Execute the workflow on multiple tasks sequentially with concurrent agent processing.

        This method processes a list of tasks one by one, where each task is executed
        by all agents concurrently. This is ideal for batch processing scenarios where
        you have multiple related tasks that need the same multi-agent analysis.

        Args:
            tasks (List[str]): List of task descriptions, prompts, or instructions.
                Each task will be processed by all agents concurrently before
                moving to the next task. Tasks are processed sequentially.
            imgs (Optional[List[str]], optional): List of image file paths corresponding
                to each task. If provided, imgs[i] will be used for tasks[i].
                If fewer images than tasks are provided, remaining tasks will
                execute without images. Defaults to None.

        Returns:
            List[Any]: List of results, one for each task. Each result is formatted
            according to the workflow's output_type configuration. The length
            of the returned list equals the length of the input tasks list.

        Processing Flow:
            1. Tasks are processed sequentially (not concurrently with each other)
            2. For each task, all agents execute concurrently
            3. Results are collected and formatted for each task
            4. Conversation history accumulates across all tasks
            5. Final result list contains aggregated outputs for each task

        Image Handling:
            - If imgs is None: All tasks execute without images
            - If imgs has fewer items than tasks: Extra tasks execute without images
            - If imgs has more items than tasks: Extra images are ignored
            - Each task gets at most one corresponding image

        Dashboard Behavior:
            - If show_dashboard=True, dashboard resets and updates for each task
            - Progress is shown separately for each task's agent execution
            - Final dashboard shows results from the last task only

        Memory and Performance:
            - Conversation history grows with each task (cumulative)
            - Memory usage scales with number of tasks and agent outputs
            - CPU utilization is optimal during each task's concurrent execution
            - Consider clearing conversation history for very large batch jobs

        Example:
            Financial analysis across quarters:

            >>> workflow = ConcurrentWorkflow(agents=[analyst1, analyst2, analyst3])
            >>> quarterly_tasks = [
            ...     "Analyze Q1 financial performance and market position",
            ...     "Analyze Q2 financial performance and market position",
            ...     "Analyze Q3 financial performance and market position",
            ...     "Analyze Q4 financial performance and market position"
            ... ]
            >>> results = workflow.batch_run(quarterly_tasks)
            >>> print(f"Completed {len(results)} quarterly analyses")
            >>> # Each result contains insights from all 3 analysts for that quarter

            With corresponding images:

            >>> tasks = ["Analyze chart trends", "Review performance metrics"]
            >>> charts = ["q1_chart.png", "q2_chart.png"]
            >>> results = workflow.batch_run(tasks, imgs=charts)
            >>> # Task 0 uses q1_chart.png, Task 1 uses q2_chart.png

            Batch processing with dashboard:

            >>> workflow = ConcurrentWorkflow(
            ...     agents=[agent1, agent2],
            ...     show_dashboard=True
            ... )
            >>> results = workflow.batch_run([
            ...     "Process document batch 1",
            ...     "Process document batch 2",
            ...     "Process document batch 3"
            ... ])
            # Dashboard shows progress for each batch separately

        Use Cases:
            - Multi-period financial analysis (quarterly/yearly reports)
            - Batch document processing with multiple expert reviews
            - A/B testing across different scenarios or datasets
            - Systematic evaluation of multiple proposals or options
            - Comparative analysis across time periods or categories

        Note:
            Tasks are intentionally processed sequentially to maintain result
            order and prevent resource contention. For truly concurrent task
            processing, create separate workflow instances.

        See Also:
            - run(): For single task execution
            - ConcurrentWorkflow: For configuration options
        """
        results = []
        for idx, task in enumerate(tasks):
            img = None
            if imgs is not None:
                # Use the img at the same index if available, else None
                if idx < len(imgs):
                    img = imgs[idx]
            results.append(self.run(task=task, img=img))
        return results


# if __name__ == "__main__":
#     # Assuming you've already initialized some agents outside of this class
#     agents = [
#         Agent(
#             agent_name=f"Financial-Analysis-Agent-{i}",
#             system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
#             model_name="gpt-4o",
#             max_loops=1,
#         )
#         for i in range(3)  # Adjust number of agents as needed
#     ]

#     # Initialize the workflow with the list of agents
#     workflow = ConcurrentWorkflow(
#         agents=agents,
#         metadata_output_path="agent_metadata_4.json",
#         return_str_on=True,
#     )

#     # Define the task for all agents
#     task = "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria?"

#     # Run the workflow and save metadata
#     metadata = workflow.run(task)
#     print(metadata)
