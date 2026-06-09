import concurrent.futures
import json
import os
import traceback
from typing import Dict, List, Optional

from swarms.agents.heavy_swarm_agents import (
    SwarmVariant,
    build_heavy_swarm_agents,
)
from swarms.prompts.heavy_swarm_prompts import (
    grok_heavy_schema,
    schema,
)
from swarms.structs.conversation import Conversation
from swarms.structs.serialization import SerializableMixin
from swarms.tools.tool_type import tool_type
from swarms.utils.formatter import formatter
from swarms.utils.heavy_swarm_dashboard import HeavySwarmDashboard
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.litellm_wrapper import LiteLLM


class HeavySwarm(SerializableMixin):
    """
        HeavySwarm is a sophisticated multi-agent orchestration system that
        decomposes complex tasks into specialized questions and executes them
        using four specialized agents: Research, Analysis, Alternatives, and
        Verification. The results are then synthesized into a comprehensive
        response.

        This swarm architecture provides robust task analysis through:
        - Intelligent question generation for specialized agent roles
        - Parallel execution of specialized agents for efficiency
        - Comprehensive synthesis of multi-perspective results
        - Real-time progress monitoring with rich dashboard displays
        - Reliability checks and validation systems
        - Multi-loop iterative refinement with context preservation

        The HeavySwarm follows a structured workflow:
        1. Task decomposition into specialized questions
        2. Parallel execution by specialized agents
        3. Result synthesis and integration
        4. Comprehensive final report generation
        5. Optional iterative refinement through multiple loops

        Key Features:
        - **Multi-loop Execution**: The max_loops parameter enables iterative
          refinement where each subsequent loop builds upon the context and
          results from previous loops
    S **Iterative Refinement**: Each loop can refine, improve, or complete
          aspects of the analysis based on previous results

        Attributes:
            name (str): Name identifier for the swarm instance
            description (str): Description of the swarm's purpose
            agents (Dict[str, Agent]): Dictionary of specialized agent instances (created internally)
            timeout (int): Maximum execution time per agent in seconds
            question_agent_model_name (str): Model name for question generation
            worker_model_name (str): Model name for specialized worker agents
            verbose (bool): Enable detailed logging output
            max_workers (int): Maximum number of concurrent worker threads
            show_dashboard (bool): Enable rich dashboard with progress visualization
            agent_prints_on (bool): Enable individual agent output printing
            max_loops (int): Maximum number of execution loops for iterative refinement
            conversation (Conversation): Conversation history tracker
            console (Console): Rich console for dashboard output

        Example:
            >>> swarm = HeavySwarm(
            ...     name="AnalysisSwarm",
            ...     description="Market analysis swarm",
            ...     question_agent_model_name="gpt-5.4",
            ...     worker_model_name="gpt-5.4",
            ...     show_dashboard=True,
            ...     max_loops=3
            ... )
            >>> result = swarm.run("Analyze the current cryptocurrency market trends")
            >>> # The swarm will run 3 iterations, each building upon the previous results
    """

    _to_dict_exclude = (
        "agents",
        "conversation",
        "dashboard",
        "worker_tools",
    )

    def __init__(
        self,
        name: str = "HeavySwarm",
        description: str = (
            "A swarm of agents that can analyze a task and generate "
            "specialized questions for each agent role"
        ),
        timeout: int = 900,
        question_agent_model_name: str = "gpt-5.4",
        worker_model_name: str = "gpt-5.4",
        verbose: bool = False,
        show_dashboard: bool = False,
        agent_prints_on: bool = False,
        output_type: str = "dict-all-except-first",
        worker_tools: Optional[tool_type] = None,
        max_loops: Optional[int] = 1,
        variant: SwarmVariant = "default",
    ) -> None:
        """
        Initialize the HeavySwarm with configuration parameters.

        Args:
            name (str, optional): Identifier name for the swarm instance. Defaults to "HeavySwarm".
            description (str, optional): Description of the swarm's purpose and capabilities.
                Defaults to standard description.
            agents (List[Agent], optional): Pre-configured agent list (currently unused as agents
                are created internally). Defaults to None.
            timeout (int, optional): Maximum execution time per agent in seconds. Defaults to 300.
            question_agent_model_name (str, optional): Language model for
                question generation. Defaults to "gpt-5.4".
            worker_model_name (str, optional): Language model for specialized
                worker agents. Defaults to "gpt-5.4".
            verbose (bool, optional): Enable detailed logging and debug output. Defaults to False.
            show_dashboard (bool, optional): Enable rich dashboard with
                progress visualization. Defaults to False.
            agent_prints_on (bool, optional): Enable individual agent
                output printing. Defaults to False.
            output_type (str, optional): Output format type for conversation
                history. Defaults to "dict-all-except-first".
            worker_tools (tool_type, optional): Tools available to worker
                agents for enhanced functionality. Defaults to None.
            max_loops (int, optional): Maximum number of execution loops for
                the entire swarm. Each loop builds upon previous results for
                iterative refinement. Defaults to 1.
            variant (Literal["default", "medium", "heavy"], optional): Swarm
                architecture to instantiate. ``"default"`` gives 5 agents
                (Research / Analysis / Alternatives / Verification / Synthesis),
                ``"medium"`` gives 4 Grok agents (Captain + Harper / Benjamin /
                Lucas), and ``"heavy"`` gives 16 agents (Grok + 15 domain
                specialists). Defaults to ``"default"``.

        Raises:
            ValueError: If required model names are None or ``variant`` is unknown.

        Note:
            The swarm automatically performs reliability checks during initialization
            to ensure all required parameters are properly configured. The max_loops
            parameter enables iterative refinement by allowing the swarm to process
            the same task multiple times, with each subsequent loop building upon
            the context and results from previous loops.
        """
        self.name = name
        self.description = description
        self.timeout = timeout
        self.question_agent_model_name = question_agent_model_name
        self.worker_model_name = worker_model_name
        self.verbose = verbose
        self.max_workers = int(os.cpu_count() * 0.9)
        self.show_dashboard = show_dashboard
        self.agent_prints_on = agent_prints_on
        self.output_type = output_type
        self.worker_tools = worker_tools
        self.max_loops = max_loops
        self.variant = variant

        self.conversation = Conversation(
            time_enabled=True, message_id_on=True
        )
        self.dashboard = HeavySwarmDashboard()

        self.agents = self.create_agents()

        if self.show_dashboard:
            self.show_swarm_info()

        self.reliability_check()

    def show_swarm_info(self):
        """
        Display comprehensive swarm configuration information in a rich dashboard format.

        This method creates and displays a professionally styled information table containing
        all key swarm configuration parameters including models, timeouts, and operational
        settings. The display uses Swarms-inspired styling with red headers and borders.

        The dashboard includes:
        - Swarm identification (name, description)
        - Execution parameters (timeout, loops per agent)
        - Model configurations (question and worker models)
        - Performance settings (max workers, aggregation strategy)

        Note:
            This method only displays output when show_dashboard is enabled. If show_dashboard
            is False, the method returns immediately without any output.

        Returns:
            None: This method only displays output and has no return value.
        """
        if not self.show_dashboard:
            return

        self.dashboard.show_config(
            name=self.name,
            description=self.description,
            timeout=self.timeout,
            question_model=self.question_agent_model_name,
            worker_model=self.worker_model_name,
            max_workers=self.max_workers,
        )

    def reliability_check(self):
        """
        Perform comprehensive reliability and configuration validation checks.

        This method validates all critical swarm configuration parameters to ensure
        the system is properly configured for operation. It checks for common
        configuration errors and provides clear error messages for any issues found.

        Validation checks include:
        - worker_model_name: Must be set for agent execution
        - question_agent_model_name: Must be set for question generation

        The method provides different user experiences based on the show_dashboard setting:
        - With dashboard: Shows animated progress bars with professional styling
        - Without dashboard: Provides basic console output with completion confirmation

        Raises:
            ValueError: If worker_model_name is None (agents can't be created)
            ValueError: If question_agent_model_name is None (questions can't be generated)

        Note:
            This method is automatically called during __init__ to ensure the swarm
            is properly configured before any operations begin.
        """
        if self.show_dashboard:
            with self.dashboard.reliability_progress(total=3) as step:
                if self.worker_model_name is None:
                    raise ValueError(
                        "worker_model_name must be set. This parameter is used to determine the model that will be used to execute the agents."
                    )
                step("[white]✓ WORKER MODEL VALIDATED")

                if self.question_agent_model_name is None:
                    raise ValueError(
                        "question_agent_model_name must be set. This parameter is used to determine the model that will be used to generate the questions."
                    )
                step("[white]✓ QUESTION MODEL VALIDATED")

                step("[bold white]✓ ALL RELIABILITY CHECKS PASSED!")

            self.dashboard.show_reliability_complete()
        else:
            # Original non-dashboard behavior
            if self.worker_model_name is None:
                raise ValueError(
                    "worker_model_name must be set. This parameter is used to determine the model that will be used to execute the agents."
                )

            if self.question_agent_model_name is None:
                raise ValueError(
                    "question_agent_model_name must be set. This parameter is used to determine the model that will be used to generate the questions."
                )

            formatter.print_panel(
                content="Reliability check passed",
                title="Reliability Check",
            )

    def run(self, task: str, img: Optional[str] = None) -> str:
        """
        Execute the complete HeavySwarm orchestration flow with multi-loop functionality.

        This method implements the max_loops feature, allowing the HeavySwarm to iterate
        multiple times on the same task, with each subsequent loop building upon the
        context and results from previous loops. This enables iterative refinement and
        deeper analysis of complex tasks.

        The method follows this workflow:
        1. For the first loop: Execute the original task with full HeavySwarm orchestration
        2. For subsequent loops: Combine previous results with original task as context
        3. Maintain conversation history across all loops for context preservation
        4. Return the final synthesized result from the last loop

        Args:
            task (str): The main task to analyze and iterate upon
            img (str, optional): Image input if needed for visual analysis tasks

        Returns:
            str: Comprehensive final answer from synthesis agent after all loops complete

        Note:
            The max_loops parameter controls how many iterations the swarm will perform.
            Each loop builds upon the previous results, enabling iterative refinement.
        """
        self._log(
            "INFO",
            f"Starting HeavySwarm: {self.name} on Task: {task}",
        )

        current_loop = 0
        last_output = None

        if self.show_dashboard:
            self.dashboard.show_task_init(task)

        # Add initial task to conversation
        self.conversation.add(
            role="User",
            content=task,
            category="input",
        )

        # Main execution loop with comprehensive error handling
        try:
            while current_loop < self.max_loops:
                try:
                    self._log("INFO", "Processing task iteration")

                    # No additional per-loop panels; keep dashboard output minimal and original-style

                    # Determine the task for this loop
                    if current_loop == 0:
                        # First loop: use the original task
                        loop_task = task
                        self._log(
                            "INFO", "First loop: Using original task"
                        )
                    else:
                        # Subsequent loops: combine previous results with original task
                        loop_task = (
                            f"Previous loop results: {last_output}\n\n"
                            f"Original task: {task}\n\n"
                            "Based on the previous results and analysis, continue with the next iteration. "
                            "Refine, improve, or complete any remaining aspects of the analysis. "
                            "Build upon the insights from the previous loop to provide deeper analysis."
                        )
                        self._log(
                            "INFO",
                            "Subsequent loop: Building upon previous results",
                        )

                    # Question generation with dashboard
                    try:
                        if self.show_dashboard:
                            with self.dashboard.question_generation_progress() as (
                                start,
                                finish,
                            ):
                                start()
                                questions = (
                                    self.execute_question_generation(
                                        loop_task
                                    )
                                )
                                finish()
                        else:
                            questions = (
                                self.execute_question_generation(
                                    loop_task
                                )
                            )

                        self.conversation.add(
                            role="Question Generator Agent",
                            content=questions,
                            category="output",
                        )

                        if "error" in questions:
                            error_msg = f"Error in question generation: {questions['error']}"
                            self._log("ERROR", error_msg)
                            return error_msg

                    except Exception as e:
                        error_msg = f"Failed to generate questions in loop {current_loop + 1}: {str(e)}"
                        self._log("ERROR", error_msg)
                        self._log(
                            "ERROR",
                            f"Traceback: {traceback.format_exc()}",
                        )
                        return error_msg

                    # Agent execution phase
                    try:
                        if self.show_dashboard:
                            agent_count = (
                                15
                                if self.variant == "heavy"
                                else (
                                    3
                                    if self.variant == "medium"
                                    else 4
                                )
                            )
                            agent_label = (
                                "Grok Heavy agents"
                                if self.variant == "heavy"
                                else (
                                    "Grok agents (Harper, "
                                    "Benjamin, Lucas)"
                                    if self.variant == "medium"
                                    else "agents"
                                )
                            )
                            self.dashboard.show_agent_launch_phase(
                                agent_count, agent_label
                            )

                        agent_results = self._execute_agents_parallel(
                            questions=questions,
                            agents=self.agents,
                            img=img,
                        )

                    except Exception as e:
                        error_msg = f"Failed to execute agents in loop {current_loop + 1}: {str(e)}"
                        self._log("ERROR", error_msg)
                        self._log(
                            "ERROR",
                            f"Traceback: {traceback.format_exc()}",
                        )
                        return error_msg

                    # Synthesis phase
                    try:
                        if self.show_dashboard:
                            synth_name = (
                                "Grok"
                                if (
                                    self.variant == "heavy"
                                    or self.variant == "medium"
                                )
                                else "Agent 5"
                            )
                            with self.dashboard.synthesis_progress(
                                synth_name
                            ) as update_stage:
                                update_stage("integrating")
                                update_stage("summarizing")
                                final_result = (
                                    self._synthesize_results(
                                        original_task=loop_task,
                                        questions=questions,
                                        agent_results=agent_results,
                                    )
                                )
                                update_stage("generating")
                                update_stage("complete")

                            self.dashboard.show_synthesis_complete()
                        else:
                            final_result = self._synthesize_results(
                                original_task=loop_task,
                                questions=questions,
                                agent_results=agent_results,
                            )

                        self.conversation.add(
                            role="Synthesis Agent",
                            content=final_result,
                            category="output",
                        )

                    except Exception as e:
                        error_msg = f"Failed to synthesize results in loop {current_loop + 1}: {str(e)}"
                        self._log("ERROR", error_msg)
                        self._log(
                            "ERROR",
                            f"Traceback: {traceback.format_exc()}",
                        )
                        return error_msg

                    # Store the result for next loop context
                    last_output = final_result
                    current_loop += 1

                    self._log(
                        "SUCCESS",
                        "Task iteration completed successfully",
                    )

                except Exception as e:
                    error_msg = f"Failed to execute loop {current_loop + 1}: {str(e)}"
                    self._log("ERROR", error_msg)
                    self._log(
                        "ERROR",
                        f"Traceback: {traceback.format_exc()}",
                    )
                    return error_msg

        except Exception as e:
            error_msg = (
                f"Critical error in HeavySwarm execution: {str(e)}"
            )
            self._log("ERROR", error_msg)
            self._log("ERROR", f"Traceback: {traceback.format_exc()}")
            return error_msg

        # Final completion message
        if self.show_dashboard:
            self.dashboard.show_synthesis_complete()

        self._log("SUCCESS", "HeavySwarm execution completed")

        return history_output_formatter(
            conversation=self.conversation,
            type=self.output_type,
        )

    def create_agents(self):
        """Build the swarm's agents.

        Returns a ``dict`` of named ``Agent`` instances; the variant
        (default / Grok / Grok Heavy) is chosen from the swarm flags.
        Called once in ``__init__`` and cached on ``self.agents``.
        """
        self._log("INFO", "🏗️ Creating specialized agents...")
        return build_heavy_swarm_agents(
            model_name=self.worker_model_name,
            tools=self.worker_tools,
            worker_prints_on=self.agent_prints_on,
            variant=self.variant,
        )

    def _execute_agents_parallel(
        self, questions: Dict, agents: Dict, img: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Execute the 4 specialized agents in TRUE parallel using concurrent.futures.

        Args:
            questions (Dict): Generated questions for each agent
            agents (Dict): Dictionary of specialized agents
            img (str, optional): Image input if needed

        Returns:
            Dict[str, str]: Results from each agent
        """

        if self.show_dashboard:
            return self._execute_agents_with_dashboard(
                questions, agents, img
            )
        else:
            return self._execute_agents_basic(questions, agents, img)

    def _execute_agents_basic(
        self, questions: Dict, agents: Dict, img: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Execute specialized agents in parallel without dashboard visualization.

        This method provides the core agent execution functionality using concurrent.futures
        for true parallel processing. It executes the four specialized agents simultaneously
        to maximize efficiency while providing basic error handling and timeout management.

        The execution process:
        1. Prepare agent tasks with their respective specialized questions
        2. Submit all tasks to ThreadPoolExecutor for parallel execution
        3. Collect results as agents complete their work
        4. Handle timeouts and exceptions gracefully
        5. Log results to conversation history

        Args:
            questions (Dict): Generated questions containing keys:
                - research_question: Question for Research Agent
                - analysis_question: Question for Analysis Agent
                - alternatives_question: Question for Alternatives Agent
                - verification_question: Question for Verification Agent
            agents (Dict): Dictionary of specialized agent instances from create_agents()
            img (str, optional): Image input for agents that support visual analysis.
                Defaults to None.

        Returns:
            Dict[str, str]: Results from each agent execution with keys:
                - 'research': Research Agent output
                - 'analysis': Analysis Agent output
                - 'alternatives': Alternatives Agent output
                - 'verification': Verification Agent output

        Note:
            This method uses ThreadPoolExecutor with max_workers limit for parallel execution.
            Each agent runs independently and results are collected as they complete.
            Timeout and exception handling ensure robustness even if individual agents fail.
        """

        # Define agent execution tasks
        def execute_agent(agent_info):
            agent_type, agent, question = agent_info
            try:
                result = agent.run(question)

                self.conversation.add(
                    role=agent.agent_name,
                    content=result,
                    category="output",
                )
                return agent_type, result
            except Exception as e:
                self._log(
                    "ERROR",
                    f"❌ Error in {agent_type} Agent: {str(e)} Traceback: {traceback.format_exc()}",
                )
                return agent_type, f"Error: {str(e)}"

        # Prepare agent tasks
        if self.variant == "heavy":
            heavy_keys = [
                ("Harper", "harper"),
                ("Benjamin", "benjamin"),
                ("Lucas", "lucas"),
                ("Olivia", "olivia"),
                ("James", "james"),
                ("Charlotte", "charlotte"),
                ("Henry", "henry"),
                ("Mia", "mia"),
                ("William", "william"),
                ("Sebastian", "sebastian"),
                ("Jack", "jack"),
                ("Owen", "owen"),
                ("Luna", "luna"),
                ("Elizabeth", "elizabeth"),
                ("Noah", "noah"),
            ]
            agent_tasks = [
                (
                    name,
                    agents[key],
                    questions.get(f"{key}_question", ""),
                )
                for name, key in heavy_keys
            ]
        elif self.variant == "medium":
            agent_tasks = [
                (
                    "Harper",
                    agents["harper"],
                    questions.get("harper_question", ""),
                ),
                (
                    "Benjamin",
                    agents["benjamin"],
                    questions.get("benjamin_question", ""),
                ),
                (
                    "Lucas",
                    agents["lucas"],
                    questions.get("lucas_question", ""),
                ),
            ]
        else:
            agent_tasks = [
                (
                    "Research",
                    agents["research"],
                    questions.get("research_question", ""),
                ),
                (
                    "Analysis",
                    agents["analysis"],
                    questions.get("analysis_question", ""),
                ),
                (
                    "Alternatives",
                    agents["alternatives"],
                    questions.get("alternatives_question", ""),
                ),
                (
                    "Verification",
                    agents["verification"],
                    questions.get("verification_question", ""),
                ),
            ]

        # Execute agents in parallel using ThreadPoolExecutor
        results = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Submit all agent tasks
            future_to_agent = {
                executor.submit(execute_agent, task): task[0]
                for task in agent_tasks
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(
                future_to_agent
            ):
                agent_type = future_to_agent[future]
                try:
                    agent_name, result = future.result(
                        timeout=self.timeout
                    )
                    results[agent_name.lower()] = result
                except concurrent.futures.TimeoutError:
                    self._log(
                        "ERROR",
                        f"⏰ Timeout for {agent_type} Agent after {self.timeout}s",
                    )
                    results[agent_type.lower()] = (
                        f"Timeout after {self.timeout} seconds"
                    )
                except Exception as e:
                    self._log(
                        "ERROR",
                        f"❌ Exception in {agent_type} Agent: {str(e)}",
                    )
                    results[agent_type.lower()] = (
                        f"Exception: {str(e)}"
                    )

        return results

    def _execute_agents_with_dashboard(
        self, questions: Dict, agents: Dict, img: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Execute specialized agents in parallel with rich dashboard visualization and progress tracking.

        This method provides an enhanced user experience by displaying real-time progress bars
        and status updates for each agent execution. It combines the efficiency of parallel
        processing with professional dashboard visualization using Rich console styling.

        Dashboard Features:
        - Individual progress bars for each of the 4 specialized agents
        - Real-time status updates with professional Swarms-inspired styling
        - Animated dots and progress indicators for visual engagement
        - Color-coded status messages (red for processing, white for completion)
        - Completion summary with mission accomplished messaging

        Progress Phases for Each Agent:
        1. INITIALIZING: Agent setup and preparation
        2. PROCESSING QUERY: Question analysis and processing
        3. EXECUTING: Core agent execution with animated indicators
        4. GENERATING RESPONSE: Response formulation and completion
        5. COMPLETE: Successful execution confirmation

        Args:
            questions (Dict): Generated specialized questions containing:
                - research_question: Comprehensive information gathering query
                - analysis_question: Pattern recognition and insight analysis query
                - alternatives_question: Creative solutions and options exploration query
                - verification_question: Validation and feasibility assessment query
            agents (Dict): Dictionary of specialized agent instances with keys:
                - research, analysis, alternatives, verification
            img (str, optional): Image input for agents supporting visual analysis.
                Defaults to None.

        Returns:
            Dict[str, str]: Comprehensive results from agent execution:
                - Keys correspond to agent types (research, analysis, alternatives, verification)
                - Values contain detailed agent outputs and analysis

        Note:
            This method requires show_dashboard=True in the HeavySwarm configuration.
            It provides the same parallel execution as _execute_agents_basic but with
            enhanced visual feedback and professional presentation.
        """

        # Agent configurations with professional styling
        if self.variant == "heavy":
            agent_configs = [
                (
                    "Harper",
                    "harper",
                    "white",
                    "Crafting narrative and storytelling angles",
                ),
                (
                    "Benjamin",
                    "benjamin",
                    "white",
                    "Analyzing data, finance and economics",
                ),
                (
                    "Lucas",
                    "lucas",
                    "white",
                    "Building code and technical solutions",
                ),
                (
                    "Olivia",
                    "olivia",
                    "white",
                    "Exploring literature, arts and culture",
                ),
                (
                    "James",
                    "james",
                    "white",
                    "Examining history, politics and philosophy",
                ),
                (
                    "Charlotte",
                    "charlotte",
                    "white",
                    "Applying math, statistics and logic",
                ),
                (
                    "Henry",
                    "henry",
                    "white",
                    "Engineering and innovation analysis",
                ),
                (
                    "Mia",
                    "mia",
                    "white",
                    "Biology, health and medicine perspective",
                ),
                (
                    "William",
                    "william",
                    "white",
                    "Business strategy and entrepreneurship",
                ),
                (
                    "Sebastian",
                    "sebastian",
                    "white",
                    "Physics, astronomy and hard sciences",
                ),
                (
                    "Jack",
                    "jack",
                    "white",
                    "Psychology and human behavior insights",
                ),
                (
                    "Owen",
                    "owen",
                    "white",
                    "Environment and global systems view",
                ),
                (
                    "Luna",
                    "luna",
                    "white",
                    "Space exploration and futurism lens",
                ),
                (
                    "Elizabeth",
                    "elizabeth",
                    "white",
                    "Ethics, policy and critical thinking",
                ),
                (
                    "Noah",
                    "noah",
                    "white",
                    "Long-term innovation and systems thinking",
                ),
            ]
        elif self.variant == "medium":
            agent_configs = [
                (
                    "Harper",
                    "harper",
                    "white",
                    "Gathering evidence and " "verifying facts",
                ),
                (
                    "Benjamin",
                    "benjamin",
                    "white",
                    "Applying rigorous logic " "and verification",
                ),
                (
                    "Lucas",
                    "lucas",
                    "white",
                    "Exploring creative angles " "and blind spots",
                ),
            ]
        else:
            agent_configs = [
                (
                    "Agent 1",
                    "research",
                    "white",
                    "Gathering comprehensive " "research data",
                ),
                (
                    "Agent 2",
                    "analysis",
                    "white",
                    "Analyzing patterns and " "generating insights",
                ),
                (
                    "Agent 3",
                    "alternatives",
                    "white",
                    "Exploring creative solutions "
                    "and alternatives",
                ),
                (
                    "Agent 4",
                    "verification",
                    "white",
                    "Validating findings and " "checking feasibility",
                ),
            ]

        results = {}

        with self.dashboard.agent_progress_tracker(
            agent_configs
        ) as tracker:

            def execute_agent_with_progress(agent_info):
                agent_type, agent_key, agent, question = agent_info
                try:
                    tracker.initializing(agent_key, agent_type)
                    tracker.processing(agent_key, agent_type)
                    tracker.executing(agent_key, agent_type)

                    result = agent.run(question)

                    tracker.responding(agent_key, agent_type)

                    self.conversation.add(
                        role=agent.agent_name,
                        content=result,
                        category="output",
                    )

                    tracker.complete(agent_key, agent_type)

                    return agent_type, result

                except Exception as e:
                    tracker.error(agent_key, agent_type)
                    self._log(
                        "ERROR",
                        f"❌ Error in {agent_type} Agent: {str(e)} Traceback: {traceback.format_exc()}",
                    )
                    return agent_type, f"Error: {str(e)}"

            # Prepare agent tasks with keys
            if self.variant == "heavy":
                heavy_keys = [
                    ("Harper", "harper"),
                    ("Benjamin", "benjamin"),
                    ("Lucas", "lucas"),
                    ("Olivia", "olivia"),
                    ("James", "james"),
                    ("Charlotte", "charlotte"),
                    ("Henry", "henry"),
                    ("Mia", "mia"),
                    ("William", "william"),
                    ("Sebastian", "sebastian"),
                    ("Jack", "jack"),
                    ("Owen", "owen"),
                    ("Luna", "luna"),
                    ("Elizabeth", "elizabeth"),
                    ("Noah", "noah"),
                ]
                agent_tasks = [
                    (
                        name,
                        key,
                        agents[key],
                        questions.get(f"{key}_question", ""),
                    )
                    for name, key in heavy_keys
                ]
            elif self.variant == "medium":
                agent_tasks = [
                    (
                        "Harper",
                        "harper",
                        agents["harper"],
                        questions.get("harper_question", ""),
                    ),
                    (
                        "Benjamin",
                        "benjamin",
                        agents["benjamin"],
                        questions.get("benjamin_question", ""),
                    ),
                    (
                        "Lucas",
                        "lucas",
                        agents["lucas"],
                        questions.get("lucas_question", ""),
                    ),
                ]
            else:
                agent_tasks = [
                    (
                        "Agent 1",
                        "research",
                        agents["research"],
                        questions.get("research_question", ""),
                    ),
                    (
                        "Agent 2",
                        "analysis",
                        agents["analysis"],
                        questions.get("analysis_question", ""),
                    ),
                    (
                        "Agent 3",
                        "alternatives",
                        agents["alternatives"],
                        questions.get(
                            "alternatives_question",
                            "",
                        ),
                    ),
                    (
                        "Agent 4",
                        "verification",
                        agents["verification"],
                        questions.get(
                            "verification_question",
                            "",
                        ),
                    ),
                ]

            # Execute agents in parallel
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                # Submit all agent tasks
                future_to_agent = {
                    executor.submit(
                        execute_agent_with_progress, task
                    ): task[1]
                    for task in agent_tasks
                }

                # Collect results as they complete
                for future in concurrent.futures.as_completed(
                    future_to_agent
                ):
                    agent_key = future_to_agent[future]
                    try:
                        agent_name, result = future.result(
                            timeout=self.timeout
                        )
                        results[
                            agent_name.lower()
                            .replace("🔍 ", "")
                            .replace("📊 ", "")
                            .replace("⚡ ", "")
                            .replace("✅ ", "")
                        ] = result
                    except concurrent.futures.TimeoutError:
                        tracker.timeout(agent_key, agent_key)
                        results[agent_key] = (
                            f"Timeout after {self.timeout} seconds"
                        )
                    except Exception as e:
                        tracker.error(agent_key, agent_key)
                        results[agent_key] = f"Exception: {str(e)}"

        # Show completion summary
        agent_count = (
            15
            if self.variant == "heavy"
            else 3 if self.variant == "medium" else 4
        )
        synth_label = (
            "Grok"
            if self.variant == "heavy"
            else (
                "Captain Swarm"
                if self.variant == "medium"
                else "synthesis"
            )
        )
        self.dashboard.show_execution_complete(
            agent_count, synth_label
        )

        return results

    def _synthesize_results(
        self, original_task: str, questions: Dict, agent_results: Dict
    ) -> str:
        """
        Synthesize all agent results into a comprehensive final answer.

        Args:
            original_task (str): The original user task
            questions (Dict): Generated questions
            agent_results (Dict): Results from all agents

        Returns:
            str: Comprehensive synthesized analysis
        """
        agents = self.agents

        if self.variant == "heavy":
            synthesis_agent = agents["captain"]
            agents_names = [
                "Harper (Creative Writing)",
                "Benjamin (Data & Finance)",
                "Lucas (Coding & Tech)",
                "Olivia (Literature & Arts)",
                "James (History & Philosophy)",
                "Charlotte (Math & Statistics)",
                "Henry (Engineering)",
                "Mia (Biology & Health)",
                "William (Business Strategy)",
                "Sebastian (Physics & Science)",
                "Jack (Psychology)",
                "Owen (Environment)",
                "Luna (Space & Futurism)",
                "Elizabeth (Ethics & Policy)",
                "Noah (Systems Thinking)",
            ]

            synthesis_prompt = f"""
        As Grok, synthesize the outputs of your 15 specialist agents into a unified, decision-grade response.

        Original Task:
        {original_task}

        Your objectives:
        - Integrate findings from all 15 specialists: {", ".join(agents_names)}
        - Identify cross-domain convergences and contradictions
        - Mediate conflicts: when specialists disagree, weigh evidence quality and explain your resolution
        - Surface genuine uncertainties rather than forcing false consensus
        - Deliver prioritized, actionable recommendations with confidence levels
        - Flag risks, ethical concerns (Elizabeth), long-term systemic issues (Noah), and mitigation strategies

        Conversation history for full context:

        \n\n

        {self.conversation.return_history_as_string()}

        \n\n

        Present your synthesis as:
        1. Executive Summary (3-5 sentences)
        2. Cross-Domain Convergences (what multiple specialists agree on)
        3. Domain-Specific Key Findings (one bullet per specialist)
        4. Conflict Resolution & Integrated Analysis
        5. Prioritized Recommendations (with confidence levels)
        6. Risks, Blind Spots & Mitigation
        7. Ethical and Long-Term Considerations (Elizabeth + Noah's synthesis)
        8. Next Steps

        Be thorough, authoritative, and balance all 15 perspectives.
        """
        elif self.variant == "medium":
            synthesis_agent = agents["captain"]
            agents_names = [
                "Harper (Research & Facts)",
                "Benjamin (Logic, Math & Code)",
                "Lucas (Creative & Divergent)",
            ]

            synthesis_prompt = f"""
        As Captain Swarm, produce a unified report from your three specialist agents. Mediate any conflicts between their outputs and deliver a coherent, decision-grade response.

        Original Task:
        {original_task}

        Your objectives:
        - Integrate findings from {", ".join(agents_names)}, highlighting how each specialist's perspective contributes.
        - Mediate conflicts: where Harper's facts conflict with Benjamin's logic or Lucas's contrarian view, weigh evidence quality and explain your resolution.
        - Surface genuine uncertainties rather than forcing false consensus.
        - Provide prioritized, actionable recommendations with confidence levels.
        - Identify risks, blind spots flagged by Lucas, and mitigation strategies.

        Conversation history for context:

        \n\n

        {self.conversation.return_history_as_string()}

        \n\n

        Present your synthesis as:
        1. Executive Summary
        2. Harper's Key Findings (facts & evidence)
        3. Benjamin's Verification (logic & validation)
        4. Lucas's Perspectives (creative & contrarian)
        5. Conflict Resolution & Integrated Analysis
        6. Prioritized Recommendations
        7. Risks, Blind Spots & Mitigation
        8. Next Steps

        Be thorough, objective, and balance all perspectives.
        """
        else:
            synthesis_agent = agents["synthesis"]
            agents_names = [
                "Research Agent",
                "Analysis Agent",
                "Alternatives Agent",
                "Verification Agent",
            ]

            synthesis_prompt = f"""
        You are an expert synthesis agent tasked with producing a clear, actionable, and executive-ready report based on the following task and the results from four specialized agents (Research, Analysis, Alternatives, Verification).

        Original Task:
        {original_task}

        Your objectives:
        - Integrate and synthesize insights from all four agents {", ".join(agents_names)}, highlighting how each contributes to the overall understanding.
        - Identify and explain key themes, patterns, and any points of agreement or disagreement across the agents' findings.
        - Provide clear, prioritized, and actionable recommendations directly addressing the original task.
        - Explicitly discuss potential risks, limitations, and propose mitigation strategies.
        - Offer practical implementation guidance and concrete next steps.
        - Ensure the report is well-structured, concise, and suitable for decision-makers (executive summary style).
        - Use bullet points, numbered lists, and section headings where appropriate for clarity and readability.

        You may reference the conversation history for additional context:

        \n\n

        {self.conversation.return_history_as_string()}

        \n\n

        Please present your synthesis in the following structure:
        1. Executive Summary
        2. Key Insights from Each Agent
        3. Integrated Analysis & Themes
        4. Actionable Recommendations
        5. Risks & Mitigation Strategies
        6. Implementation Guidance & Next Steps

        Be thorough, objective, and ensure your synthesis is easy to follow for a non-technical audience.
        """

        return synthesis_agent.run(synthesis_prompt)

    def _parse_tool_calls(self, tool_calls: List) -> Dict[str, any]:
        """
        Parse ChatCompletionMessageToolCall objects into a structured dictionary format.

        This method extracts and structures the question generation results from language model
        tool calls. It handles the JSON parsing of function arguments and provides clean access
        to the generated questions for each specialized agent role.

        The method specifically looks for the 'generate_specialized_questions' function call
        and extracts the four specialized questions along with metadata. It provides robust
        error handling for JSON parsing failures and includes both successful and error cases.

        Args:
            tool_calls (List): List of ChatCompletionMessageToolCall objects returned by the LLM.
                Expected to contain at least one tool call with question generation results.

        Returns:
            Dict[str, any]: Structured dictionary containing:
                On success:
                - thinking (str): Reasoning process for question decomposition
                - research_question (str): Question for Research Agent
                - analysis_question (str): Question for Analysis Agent
                - alternatives_question (str): Question for Alternatives Agent
                - verification_question (str): Question for Verification Agent
                - tool_call_id (str): Unique identifier for the tool call
                - function_name (str): Name of the called function

                On error:
                - error (str): Error message describing the parsing failure
                - raw_arguments (str): Original unparsed function arguments
                - tool_call_id (str): Tool call identifier for debugging
                - function_name (str): Function name for debugging

        Note:
            If no tool calls are provided, returns an empty dictionary.
            Only the first tool call is processed, as only one question generation
            call is expected per task.
        """
        if not tool_calls:
            return {}

        # Get the first tool call (should be the question generation)
        tool_call = tool_calls[0]

        try:
            # Parse the JSON arguments
            arguments = json.loads(tool_call.function.arguments)

            result = dict(arguments)
            result["tool_call_id"] = tool_call.id
            result["function_name"] = tool_call.function.name
            return result

        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to parse tool call arguments: {str(e)}",
                "raw_arguments": tool_call.function.arguments,
                "tool_call_id": tool_call.id,
                "function_name": tool_call.function.name,
            }

    def execute_question_generation(
        self, task: str
    ) -> Dict[str, str]:
        """
        Execute the question generation using the schema with a language model.

        Args:
            task (str): The main task to analyze

        Returns:
            Dict[str, str]: Generated questions for each agent role with parsed data
        """

        # Create the prompt for question generation
        if self.variant == "heavy":
            prompt = f"""
        System: Grok task decomposer. Generate 15 non-overlapping domain-specific questions via function tool.

        Specialists:
        - Harper (Creative Writing & Storytelling): narrative framing, metaphors, human storytelling angles
        - Benjamin (Data, Finance & Economics): quantitative data, financial modeling, economic implications
        - Lucas (Coding, Programming & Technical): technical implementation, algorithms, systems architecture
        - Olivia (Literature, Arts & Culture): cultural context, humanistic interpretation, aesthetic dimensions
        - James (History, Politics & Philosophy): historical precedent, power dynamics, philosophical frameworks
        - Charlotte (Math, Statistics & Logic): formal reasoning, quantitative analysis, statistical inference
        - Henry (Engineering, Robotics & Innovation): physical systems, engineering trade-offs, innovation pathways
        - Mia (Biology, Health & Medicine): biological implications, health impacts, medical evidence
        - William (Business Strategy & Entrepreneurship): market strategy, competitive dynamics, business models
        - Sebastian (Physics, Astronomy & Hard Sciences): fundamental science, physical constraints, scientific consensus
        - Jack (Psychology & Human Behavior): cognitive biases, behavioral economics, psychological drivers
        - Owen (Environment, Sustainability & Global Systems): ecological impact, systemic sustainability, global effects
        - Luna (Space Exploration & Futurism): long-term futures, emerging technologies, civilizational scale
        - Elizabeth (Ethics, Policy & Critical Thinking): ethical implications, policy trade-offs, unintended consequences
        - Noah (Long-Term Innovation & Systems): systems thinking, long-term trajectories, second-order effects

        Requirements:
        - Each question ≤40 words, domain-specific, action-oriented
        - No duplication across specialists
        - Ambiguity notes only in "thinking" field (≤60 words)
        - Each question must leverage the specialist's unique domain expertise

        Task: {task}

        Use generate_grok_heavy_questions function only.
        """
            active_schema = grok_heavy_schema
        elif self.variant == "medium":
            prompt = f"""
        System: Technical task analyzer. Generate 4 non-overlapping analytical questions via function tool.

        Roles:
        - Research: systematic evidence collection, source verification, data quality assessment
        - Analysis: statistical analysis, pattern recognition, quantitative insights, correlation analysis
        - Alternatives: strategic option generation, multi-criteria analysis, scenario planning, decision modeling
        - Verification: systematic validation, risk assessment, feasibility analysis, logical consistency

        Requirements:
        - Each question ≤30 words, technically precise, action-oriented
        - No duplication across roles. No meta text in questions
        - Ambiguity notes only in "thinking" field (≤40 words)
        - Focus on systematic methodology and quantitative analysis

        Task: {task}

        Use generate_specialized_questions function only.
        """
            active_schema = schema

        max_tokens = 5000 if self.variant == "heavy" else 3000

        question_agent = LiteLLM(
            system_prompt=prompt,
            model=self.question_agent_model_name,
            tools_list_dictionary=active_schema,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            tool_choice="auto",
        )

        # Get raw tool calls from LiteLLM
        raw_output = question_agent.run(task)

        # Parse the tool calls and return clean data
        out = self._parse_tool_calls(raw_output)

        self._log(
            "INFO",
            f"🔍 Question Generation Output: {out} and type: {type(out)}",
        )

        return out

    def get_questions_only(self, task: str) -> Dict[str, str]:
        """
        Generate and extract only the specialized questions without metadata or execution.

        This utility method provides a clean interface for obtaining just the generated
        questions for each agent role without executing the full swarm workflow. It's
        useful for previewing questions, debugging question generation, or integrating
        with external systems that only need the questions.

        The method performs question generation using the configured question agent model
        and returns a clean dictionary containing only the four specialized questions,
        filtering out metadata like thinking process, tool call IDs, and function names.

        Args:
            task (str): The main task or query to analyze and decompose into specialized
                questions. Should be a clear, specific task description.

        Returns:
            Dict[str, str]: Clean dictionary containing only the questions:
                - research_question (str): Question for comprehensive information gathering
                - analysis_question (str): Question for pattern analysis and insights
                - alternatives_question (str): Question for exploring creative solutions
                - verification_question (str): Question for validation and feasibility

                On error:
                - error (str): Error message if question generation fails

        Example:
            >>> swarm = HeavySwarm()
            >>> questions = swarm.get_questions_only("Analyze market trends for EVs")
            >>> print(questions['research_question'])
        """
        result = self.execute_question_generation(task)

        if "error" in result:
            return {"error": result["error"]}

        if self.variant == "heavy":
            heavy_keys = [
                "harper",
                "benjamin",
                "lucas",
                "olivia",
                "james",
                "charlotte",
                "henry",
                "mia",
                "william",
                "sebastian",
                "jack",
                "owen",
                "luna",
                "elizabeth",
                "noah",
            ]
            return {
                f"{k}_question": result.get(f"{k}_question", "")
                for k in heavy_keys
            }

        if self.variant == "medium":
            return {
                "harper_question": result.get("harper_question", ""),
                "benjamin_question": result.get(
                    "benjamin_question", ""
                ),
                "lucas_question": result.get("lucas_question", ""),
            }

        return {
            "research_question": result.get("research_question", ""),
            "analysis_question": result.get("analysis_question", ""),
            "alternatives_question": result.get(
                "alternatives_question", ""
            ),
            "verification_question": result.get(
                "verification_question", ""
            ),
        }

    def get_questions_as_list(self, task: str) -> List[str]:
        """
        Generate specialized questions and return them as an ordered list.

        This utility method provides the simplest interface for obtaining generated questions
        in a list format. It's particularly useful for iteration, display purposes, or
        integration with systems that prefer list-based data structures over dictionaries.

        The questions are returned in a consistent order:
        1. Research question (information gathering)
        2. Analysis question (pattern recognition and insights)
        3. Alternatives question (creative solutions exploration)
        4. Verification question (validation and feasibility)

        Args:
            task (str): The main task or query to decompose into specialized questions.
                Should be a clear, actionable task description that can be analyzed
                from multiple perspectives.

        Returns:
            List[str]: Ordered list of 4 specialized questions:
                [0] Research question for comprehensive information gathering
                [1] Analysis question for pattern analysis and insights
                [2] Alternatives question for exploring creative solutions
                [3] Verification question for validation and feasibility assessment

                On error: Single-item list containing error message

        Example:
            >>> swarm = HeavySwarm()
            >>> questions = swarm.get_questions_as_list("Optimize supply chain efficiency")
            >>> for i, question in enumerate(questions):
            ...     print(f"Agent {i+1}: {question}")

        Note:
            This method internally calls get_questions_only() and converts the dictionary
            to a list format, maintaining the standard agent order.
        """
        questions = self.get_questions_only(task)

        if "error" in questions:
            return [f"Error: {questions['error']}"]

        if self.variant == "heavy":
            heavy_keys = [
                "harper",
                "benjamin",
                "lucas",
                "olivia",
                "james",
                "charlotte",
                "henry",
                "mia",
                "william",
                "sebastian",
                "jack",
                "owen",
                "luna",
                "elizabeth",
                "noah",
            ]
            return [
                questions.get(f"{k}_question", "") for k in heavy_keys
            ]

        if self.variant == "medium":
            return [
                questions.get("harper_question", ""),
                questions.get("benjamin_question", ""),
                questions.get("lucas_question", ""),
            ]

        return [
            questions.get("research_question", ""),
            questions.get("analysis_question", ""),
            questions.get("alternatives_question", ""),
            questions.get("verification_question", ""),
        ]
