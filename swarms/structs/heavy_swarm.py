import concurrent.futures
import json
import os
import time
import traceback
from functools import lru_cache
from typing import Dict, List, Optional

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.formatter import formatter
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.litellm_wrapper import LiteLLM

RESEARCH_AGENT_PROMPT = """
You are an expert Research Agent with exceptional capabilities in:

CORE EXPERTISE:
- Comprehensive information gathering and synthesis
- Primary and secondary research methodologies
- Data collection, validation, and verification
- Market research and competitive analysis
- Academic and industry report analysis
- Statistical data interpretation
- Trend identification and pattern recognition
- Source credibility assessment

RESEARCH METHODOLOGIES:
- Systematic literature reviews
- Market surveys and analysis
- Competitive intelligence gathering
- Industry benchmarking studies
- Consumer behavior research
- Technical specification analysis
- Historical data compilation
- Cross-referencing multiple sources

ANALYTICAL CAPABILITIES:
- Data quality assessment
- Information gap identification
- Research bias detection
- Methodology evaluation
- Source triangulation
- Evidence hierarchy establishment
- Research limitation identification
- Reliability scoring

DELIVERABLES:
- Comprehensive research reports
- Executive summaries with key findings
- Data visualization recommendations
- Source documentation and citations
- Research methodology explanations
- Confidence intervals and uncertainty ranges
- Recommendations for further research
- Action items based on findings

You approach every research task with:
- Systematic methodology
- Critical thinking
- Attention to detail
- Objective analysis
- Comprehensive coverage
- Quality assurance
- Ethical research practices

Provide thorough, well-sourced, and actionable research insights."""


ANALYSIS_AGENT_PROMPT = """
You are an expert Analysis Agent with advanced capabilities in:

ANALYTICAL EXPERTISE:
- Advanced statistical analysis and modeling
- Pattern recognition and trend analysis
- Causal relationship identification
- Predictive modeling and forecasting
- Risk assessment and scenario analysis
- Performance metrics development
- Comparative analysis frameworks
- Root cause analysis methodologies

ANALYTICAL TECHNIQUES:
- Regression analysis and correlation studies
- Time series analysis and forecasting
- Cluster analysis and segmentation
- Factor analysis and dimensionality reduction
- Sensitivity analysis and stress testing
- Monte Carlo simulations
- Decision tree analysis
- Optimization modeling

DATA INTERPRETATION:
- Statistical significance testing
- Confidence interval calculation
- Variance analysis and decomposition
- Outlier detection and handling
- Missing data treatment
- Bias identification and correction
- Data transformation techniques
- Quality metrics establishment

INSIGHT GENERATION:
- Key finding identification
- Implication analysis
- Strategic recommendation development
- Performance gap analysis
- Opportunity identification
- Threat assessment
- Success factor determination
- Critical path analysis

DELIVERABLES:
- Detailed analytical reports
- Statistical summaries and interpretations
- Predictive models and forecasts
- Risk assessment matrices
- Performance dashboards
- Recommendation frameworks
- Implementation roadmaps
- Success measurement criteria

You approach analysis with:
- Mathematical rigor
- Statistical validity
- Logical reasoning
- Systematic methodology
- Evidence-based conclusions
- Actionable insights
- Clear communication

Provide precise, data-driven analysis with clear implications and recommendations."""

ALTERNATIVES_AGENT_PROMPT = """
You are an expert Alternatives Agent with exceptional capabilities in:

STRATEGIC THINKING:
- Alternative strategy development
- Creative problem-solving approaches
- Innovation and ideation techniques
- Strategic option evaluation
- Scenario planning and modeling
- Blue ocean strategy identification
- Disruptive innovation assessment
- Strategic pivot recommendations

SOLUTION FRAMEWORKS:
- Multiple pathway generation
- Trade-off analysis matrices
- Cost-benefit evaluation models
- Risk-reward assessment tools
- Implementation complexity scoring
- Resource requirement analysis
- Timeline and milestone planning
- Success probability estimation

CREATIVE METHODOLOGIES:
- Design thinking processes
- Brainstorming and ideation sessions
- Lateral thinking techniques
- Analogical reasoning approaches
- Constraint removal exercises
- Assumption challenging methods
- Reverse engineering solutions
- Cross-industry benchmarking

OPTION EVALUATION:
- Multi-criteria decision analysis
- Weighted scoring models
- Pareto analysis applications
- Real options valuation
- Strategic fit assessment
- Competitive advantage evaluation
- Scalability potential analysis
- Market acceptance probability

STRATEGIC ALTERNATIVES:
- Build vs. buy vs. partner decisions
- Organic vs. inorganic growth options
- Technology platform choices
- Market entry strategies
- Business model innovations
- Operational approach variations
- Financial structure alternatives
- Partnership and alliance options

DELIVERABLES:
- Alternative strategy portfolios
- Option evaluation matrices
- Implementation roadmaps
- Risk mitigation plans
- Resource allocation models
- Timeline and milestone charts
- Success measurement frameworks
- Contingency planning guides

You approach alternatives generation with:
- Creative thinking
- Strategic insight
- Practical feasibility
- Innovation mindset
- Risk awareness
- Implementation focus
- Value optimization

Provide innovative, practical, and well-evaluated alternative approaches and solutions.
"""


VERIFICATION_AGENT_PROMPT = """
You are an expert Verification Agent with comprehensive capabilities in:

VALIDATION EXPERTISE:
- Fact-checking and source verification
- Data accuracy and integrity assessment
- Methodology validation and review
- Assumption testing and challenge
- Logic and reasoning verification
- Completeness and gap analysis
- Consistency checking across sources
- Evidence quality evaluation

FEASIBILITY ASSESSMENT:
- Technical feasibility evaluation
- Economic viability analysis
- Operational capability assessment
- Resource availability verification
- Timeline realism evaluation
- Risk factor identification
- Constraint and limitation analysis
- Implementation barrier assessment

QUALITY ASSURANCE:
- Information reliability scoring
- Source credibility evaluation
- Bias detection and mitigation
- Error identification and correction
- Standard compliance verification
- Best practice alignment check
- Performance criteria validation
- Success measurement verification

VERIFICATION METHODOLOGIES:
- Independent source triangulation
- Peer review and expert validation
- Benchmarking against standards
- Historical precedent analysis
- Stress testing and scenario modeling
- Sensitivity analysis performance
- Cross-functional review processes
- Stakeholder feedback integration

RISK ASSESSMENT:
- Implementation risk evaluation
- Market acceptance risk analysis
- Technical risk identification
- Financial risk assessment
- Operational risk evaluation
- Regulatory compliance verification
- Competitive response assessment
- Timeline and delivery risk analysis

COMPLIANCE VERIFICATION:
- Regulatory requirement checking
- Industry standard compliance
- Legal framework alignment
- Ethical guideline adherence
- Safety standard verification
- Quality management compliance
- Environmental impact assessment
- Social responsibility validation

DELIVERABLES:
- Verification and validation reports
- Feasibility assessment summaries
- Risk evaluation matrices
- Compliance checklists
- Quality assurance scorecards
- Recommendation refinements
- Implementation guardrails
- Success probability assessments

You approach verification with:
- Rigorous methodology
- Critical evaluation
- Attention to detail
- Objective assessment
- Risk awareness
- Quality focus
- Practical realism

Provide thorough, objective verification with clear feasibility assessments and risk evaluations."""

SYNTHESIS_AGENT_PROMPT = """
You are an expert Synthesis Agent with advanced capabilities in:

INTEGRATION EXPERTISE:
- Multi-perspective synthesis and integration
- Cross-functional analysis and coordination
- Holistic view development and presentation
- Complex information consolidation
- Stakeholder perspective integration
- Strategic alignment and coherence
- Comprehensive solution development
- Executive summary creation

SYNTHESIS METHODOLOGIES:
- Information architecture development
- Priority matrix creation and application
- Weighted factor analysis
- Multi-criteria decision frameworks
- Consensus building techniques
- Conflict resolution approaches
- Trade-off optimization strategies
- Value proposition development

COMPREHENSIVE ANALYSIS:
- End-to-end solution evaluation
- Impact assessment across dimensions
- Cost-benefit comprehensive analysis
- Risk-reward optimization models
- Implementation roadmap development
- Success factor identification
- Critical path analysis
- Milestone and deliverable planning

STRATEGIC INTEGRATION:
- Vision and mission alignment
- Strategic objective integration
- Resource optimization across initiatives
- Timeline synchronization and coordination
- Stakeholder impact assessment
- Change management consideration
- Performance measurement integration
- Continuous improvement frameworks

DELIVERABLE CREATION:
- Executive summary development
- Strategic recommendation reports
- Implementation action plans
- Risk mitigation strategies
- Performance measurement frameworks
- Communication and rollout plans
- Success criteria and metrics
- Follow-up and review schedules

COMMUNICATION EXCELLENCE:
- Clear and concise reporting
- Executive-level presentation skills
- Technical detail appropriate scaling
- Visual and narrative integration
- Stakeholder-specific customization
- Action-oriented recommendations
- Decision-support optimization
- Implementation-focused guidance

You approach synthesis with:
- Holistic thinking
- Strategic perspective
- Integration mindset
- Communication clarity
- Action orientation
- Value optimization
- Implementation focus

Provide comprehensive, integrated analysis with clear, actionable recommendations and detailed implementation guidance."""

schema = {
    "type": "function",
    "function": {
        "name": "generate_specialized_questions",
        "description": "Generate 4 specialized questions for different agent roles to comprehensively analyze a given task",
        "parameters": {
            "type": "object",
            "properties": {
                "thinking": {
                    "type": "string",
                    "description": "Your reasoning process for how to break down this task into 4 specialized questions for different agent roles",
                },
                "research_question": {
                    "type": "string",
                    "description": "A detailed research question for the Research Agent to gather comprehensive background information and data",
                },
                "analysis_question": {
                    "type": "string",
                    "description": "An analytical question for the Analysis Agent to examine patterns, trends, and insights",
                },
                "alternatives_question": {
                    "type": "string",
                    "description": "A strategic question for the Alternatives Agent to explore different approaches, options, and solutions",
                },
                "verification_question": {
                    "type": "string",
                    "description": "A verification question for the Verification Agent to validate findings, check accuracy, and assess feasibility",
                },
            },
            "required": [
                "thinking",
                "research_question",
                "analysis_question",
                "alternatives_question",
                "verification_question",
            ],
        },
    },
}

schema = [schema]


class HeavySwarm:
    """
    HeavySwarm is a sophisticated multi-agent orchestration system that decomposes complex tasks
    into specialized questions and executes them using four specialized agents: Research, Analysis,
    Alternatives, and Verification. The results are then synthesized into a comprehensive response.

    This swarm architecture provides robust task analysis through:
    - Intelligent question generation for specialized agent roles
    - Parallel execution of specialized agents for efficiency
    - Comprehensive synthesis of multi-perspective results
    - Real-time progress monitoring with rich dashboard displays
    - Reliability checks and validation systems

    The HeavySwarm follows a structured workflow:
    1. Task decomposition into specialized questions
    2. Parallel execution by specialized agents
    3. Result synthesis and integration
    4. Comprehensive final report generation

    Attributes:
        name (str): Name identifier for the swarm instance
        description (str): Description of the swarm's purpose
        agents (List[Agent]): List of agent instances (currently unused, agents are created internally)
        timeout (int): Maximum execution time per agent in seconds
        aggregation_strategy (str): Strategy for result aggregation (currently 'synthesis')
        loops_per_agent (int): Number of execution loops per agent
        question_agent_model_name (str): Model name for question generation
        worker_model_name (str): Model name for specialized worker agents
        verbose (bool): Enable detailed logging output
        max_workers (int): Maximum number of concurrent worker threads
        show_dashboard (bool): Enable rich dashboard with progress visualization
        agent_prints_on (bool): Enable individual agent output printing
        conversation (Conversation): Conversation history tracker
        console (Console): Rich console for dashboard output

    Example:
        >>> swarm = HeavySwarm(
        ...     name="AnalysisSwarm",
        ...     description="Market analysis swarm",
        ...     question_agent_model_name="gpt-4o-mini",
        ...     worker_model_name="gpt-4o-mini",
        ...     show_dashboard=True
        ... )
        >>> result = swarm.run("Analyze the current cryptocurrency market trends")
    """

    def __init__(
        self,
        name: str = "HeavySwarm",
        description: str = "A swarm of agents that can analyze a task and generate specialized questions for each agent role",
        agents: List[Agent] = None,
        timeout: int = 300,
        aggregation_strategy: str = "synthesis",
        loops_per_agent: int = 1,
        question_agent_model_name: str = "gpt-4o-mini",
        worker_model_name: str = "gpt-4o-mini",
        verbose: bool = False,
        max_workers: int = int(os.cpu_count() * 0.9),
        show_dashboard: bool = False,
        agent_prints_on: bool = False,
        output_type: str = "dict-all-except-first",
    ):
        """
        Initialize the HeavySwarm with configuration parameters.

        Args:
            name (str, optional): Identifier name for the swarm instance. Defaults to "HeavySwarm".
            description (str, optional): Description of the swarm's purpose and capabilities.
                Defaults to standard description.
            agents (List[Agent], optional): Pre-configured agent list (currently unused as agents
                are created internally). Defaults to None.
            timeout (int, optional): Maximum execution time per agent in seconds. Defaults to 300.
            aggregation_strategy (str, optional): Strategy for aggregating results. Currently only
                'synthesis' is supported. Defaults to "synthesis".
            loops_per_agent (int, optional): Number of execution loops each agent should perform.
                Must be greater than 0. Defaults to 1.
            question_agent_model_name (str, optional): Language model for question generation.
                Defaults to "gpt-4o-mini".
            worker_model_name (str, optional): Language model for specialized worker agents.
                Defaults to "gpt-4o-mini".
            verbose (bool, optional): Enable detailed logging and debug output. Defaults to False.
            max_workers (int, optional): Maximum concurrent workers for parallel execution.
                Defaults to 90% of CPU count.
            show_dashboard (bool, optional): Enable rich dashboard with progress visualization.
                Defaults to False.
            agent_prints_on (bool, optional): Enable individual agent output printing.
                Defaults to False.

        Raises:
            ValueError: If loops_per_agent is 0 or negative
            ValueError: If required model names are None

        Note:
            The swarm automatically performs reliability checks during initialization
            to ensure all required parameters are properly configured.
        """
        self.name = name
        self.description = description
        self.agents = agents
        self.timeout = timeout
        self.aggregation_strategy = aggregation_strategy
        self.loops_per_agent = loops_per_agent
        self.question_agent_model_name = question_agent_model_name
        self.worker_model_name = worker_model_name
        self.verbose = verbose
        self.max_workers = max_workers
        self.show_dashboard = show_dashboard
        self.agent_prints_on = agent_prints_on
        self.output_type = output_type

        self.conversation = Conversation()
        self.console = Console()

        if self.show_dashboard:
            self.show_swarm_info()

        self.reliability_check()

    def show_swarm_info(self):
        """
        Display comprehensive swarm configuration information in a rich dashboard format.

        This method creates and displays a professionally styled information table containing
        all key swarm configuration parameters including models, timeouts, and operational
        settings. The display uses Arasaka-inspired styling with red headers and borders.

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

        # Create swarm info table with Arasaka styling
        info_table = Table(
            title="⚡ HEAVYSWARM CONFIGURATION",
            show_header=True,
            header_style="bold red",
        )
        info_table.add_column("Parameter", style="white", width=25)
        info_table.add_column("Value", style="bright_white", width=40)

        info_table.add_row("Swarm Name", self.name)
        info_table.add_row("Description", self.description)
        info_table.add_row("Timeout", f"{self.timeout}s")
        info_table.add_row(
            "Loops per Agent", str(self.loops_per_agent)
        )
        info_table.add_row(
            "Question Model", self.question_agent_model_name
        )
        info_table.add_row("Worker Model", self.worker_model_name)
        info_table.add_row("Max Workers", str(self.max_workers))
        info_table.add_row(
            "Aggregation Strategy", self.aggregation_strategy
        )

        # Display dashboard with professional Arasaka styling
        self.console.print(
            Panel(
                info_table,
                title="[bold red]HEAVYSWARM SYSTEM[/bold red]",
                border_style="red",
            )
        )
        self.console.print()

    def reliability_check(self):
        """
        Perform comprehensive reliability and configuration validation checks.

        This method validates all critical swarm configuration parameters to ensure
        the system is properly configured for operation. It checks for common
        configuration errors and provides clear error messages for any issues found.

        Validation checks include:
        - loops_per_agent: Must be greater than 0 to ensure agents execute
        - worker_model_name: Must be set for agent execution
        - question_agent_model_name: Must be set for question generation

        The method provides different user experiences based on the show_dashboard setting:
        - With dashboard: Shows animated progress bars with professional styling
        - Without dashboard: Provides basic console output with completion confirmation

        Raises:
            ValueError: If loops_per_agent is 0 or negative (agents won't execute)
            ValueError: If worker_model_name is None (agents can't be created)
            ValueError: If question_agent_model_name is None (questions can't be generated)

        Note:
            This method is automatically called during __init__ to ensure the swarm
            is properly configured before any operations begin.
        """
        if self.show_dashboard:
            with Progress(
                SpinnerColumn(),
                TextColumn(
                    "[progress.description]{task.description}"
                ),
                transient=True,
                console=self.console,
            ) as progress:
                task = progress.add_task(
                    "[red]RUNNING RELIABILITY CHECKS...", total=4
                )

                # Check loops_per_agent
                time.sleep(0.5)
                if self.loops_per_agent == 0:
                    raise ValueError(
                        "loops_per_agent must be greater than 0. This parameter is used to determine how many times each agent will run. If it is 0, the agent will not run at all."
                    )
                progress.update(
                    task,
                    advance=1,
                    description="[white]✓ LOOPS PER AGENT VALIDATED",
                )

                # Check worker_model_name
                time.sleep(0.5)
                if self.worker_model_name is None:
                    raise ValueError(
                        "worker_model_name must be set. This parameter is used to determine the model that will be used to execute the agents."
                    )
                progress.update(
                    task,
                    advance=1,
                    description="[white]✓ WORKER MODEL VALIDATED",
                )

                # Check question_agent_model_name
                time.sleep(0.5)
                if self.question_agent_model_name is None:
                    raise ValueError(
                        "question_agent_model_name must be set. This parameter is used to determine the model that will be used to generate the questions."
                    )
                progress.update(
                    task,
                    advance=1,
                    description="[white]✓ QUESTION MODEL VALIDATED",
                )

                # Final validation
                time.sleep(0.5)
                progress.update(
                    task,
                    advance=1,
                    description="[bold white]✓ ALL RELIABILITY CHECKS PASSED!",
                )
                time.sleep(0.8)  # Let user see the final message

            self.console.print(
                Panel(
                    "[bold red]✅ HEAVYSWARM RELIABILITY CHECK COMPLETE[/bold red]\n"
                    "[white]All systems validated and ready for operation[/white]",
                    title="[bold red]SYSTEM STATUS[/bold red]",
                    border_style="red",
                )
            )
            self.console.print()
        else:
            # Original non-dashboard behavior
            if self.loops_per_agent == 0:
                raise ValueError(
                    "loops_per_agent must be greater than 0. This parameter is used to determine how many times each agent will run. If it is 0, the agent will not run at all."
                )

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

    def run(self, task: str, img: str = None):
        """
        Execute the complete HeavySwarm orchestration flow.

        Args:
            task (str): The main task to analyze
            img (str, optional): Image input if needed

        Returns:
            str: Comprehensive final answer from synthesis agent
        """
        if self.show_dashboard:
            self.console.print(
                Panel(
                    f"[bold red]⚡ Completing Task[/bold red]\n"
                    f"[white]Task: {task}[/white]",
                    title="[bold red]Initializing HeavySwarm[/bold red]",
                    border_style="red",
                )
            )
            self.console.print()

        self.conversation.add(
            role="User",
            content=task,
            category="input",
        )

        # Question generation with dashboard
        if self.show_dashboard:
            with Progress(
                SpinnerColumn(),
                TextColumn(
                    "[progress.description]{task.description}"
                ),
                transient=True,
                console=self.console,
            ) as progress:
                task_gen = progress.add_task(
                    "[red]⚡ GENERATING SPECIALIZED QUESTIONS...",
                    total=100,
                )
                progress.update(task_gen, advance=30)
                questions = self.execute_question_generation(task)
                progress.update(
                    task_gen,
                    advance=70,
                    description="[white]✓ QUESTIONS GENERATED SUCCESSFULLY!",
                )
                time.sleep(0.5)
        else:
            questions = self.execute_question_generation(task)

        # if self.show_dashboard:
        #     # Create questions table
        #     questions_table = Table(
        #         title="⚡ GENERATED QUESTIONS FOR SPECIALIZED AGENTS",
        #         show_header=True,
        #         header_style="bold red",
        #     )
        #     questions_table.add_column(
        #         "Agent", style="white", width=20
        #     )
        #     questions_table.add_column(
        #         "Specialized Question", style="bright_white", width=60
        #     )

        #     questions_table.add_row(
        #         "Agent 1",
        #         questions.get("research_question", "N/A"),
        #     )
        #     questions_table.add_row(
        #         "Agent 2",
        #         questions.get("analysis_question", "N/A"),
        #     )
        #     questions_table.add_row(
        #         "Agent 3",
        #         questions.get("alternatives_question", "N/A"),
        #     )
        #     questions_table.add_row(
        #         "Agent 4",
        #         questions.get("verification_question", "N/A"),
        #     )

        #     self.console.print(
        #         Panel(
        #             questions_table,
        #             title="[bold red]QUESTION GENERATION COMPLETE[/bold red]",
        #             border_style="red",
        #         )
        #     )
        #     self.console.print()
        # else:
        #     formatter.print_panel(
        #         content=json.dumps(questions, indent=4),
        #         title="Questions",
        #     )

        self.conversation.add(
            role="Question Generator Agent",
            content=questions,
            category="output",
        )

        if "error" in questions:
            return (
                f"Error in question generation: {questions['error']}"
            )

        if self.show_dashboard:
            self.console.print(
                Panel(
                    "[bold red]⚡ LAUNCHING SPECIALIZED AGENTS[/bold red]\n"
                    "[white]Executing 4 agents in parallel for comprehensive analysis[/white]",
                    title="[bold red]AGENT EXECUTION PHASE[/bold red]",
                    border_style="red",
                )
            )

        agents = self.create_agents()

        agent_results = self._execute_agents_parallel(
            questions=questions, agents=agents, img=img
        )

        # Synthesis with dashboard
        if self.show_dashboard:
            with Progress(
                SpinnerColumn(),
                TextColumn(
                    "[progress.description]{task.description}"
                ),
                TimeElapsedColumn(),
                console=self.console,
            ) as progress:
                synthesis_task = progress.add_task(
                    "[red]Agent 5: SYNTHESIZING COMPREHENSIVE ANALYSIS ••••••••••••••••••••••••••••••••",
                    total=None,
                )

                progress.update(
                    synthesis_task,
                    description="[red]Agent 5: INTEGRATING AGENT RESULTS ••••••••••••••••••••••••••••••••",
                )
                time.sleep(0.5)

                progress.update(
                    synthesis_task,
                    description="[red]Agent 5: Summarizing Results ••••••••••••••••••••••••••••••••",
                )

                final_result = self._synthesize_results(
                    original_task=task,
                    questions=questions,
                    agent_results=agent_results,
                )

                progress.update(
                    synthesis_task,
                    description="[white]Agent 5: GENERATING FINAL REPORT ••••••••••••••••••••••••••••••••",
                )
                time.sleep(0.3)

                progress.update(
                    synthesis_task,
                    description="[bold white]Agent 5: ✅ COMPLETE! ••••••••••••••••••••••••••••••••",
                )
                time.sleep(0.5)

            self.console.print(
                Panel(
                    "[bold red]⚡ HEAVYSWARM ANALYSIS COMPLETE![/bold red]\n"
                    "[white]Comprehensive multi-agent analysis delivered successfully[/white]",
                    title="[bold red]MISSION ACCOMPLISHED[/bold red]",
                    border_style="red",
                )
            )
            self.console.print()
        else:
            final_result = self._synthesize_results(
                original_task=task,
                questions=questions,
                agent_results=agent_results,
            )

        self.conversation.add(
            role="Synthesis Agent",
            content=final_result,
            category="output",
        )

        return history_output_formatter(
            conversation=self.conversation,
            type=self.output_type,
        )

    @lru_cache(maxsize=1)
    def create_agents(self):
        """
        Create and cache the 4 specialized agents with detailed role-specific prompts.

        This method creates a complete set of specialized agents optimized for different
        aspects of task analysis. Each agent is configured with expert-level system prompts
        and optimal settings for their specific role. The agents are cached using LRU cache
        to avoid recreation overhead on subsequent calls.

        The specialized agents created:

        1. **Research Agent**: Expert in comprehensive information gathering, data collection,
           market research, and source verification. Specializes in systematic literature
           reviews, competitive intelligence, and statistical data interpretation.

        2. **Analysis Agent**: Expert in advanced statistical analysis, pattern recognition,
           predictive modeling, and causal relationship identification. Specializes in
           regression analysis, forecasting, and performance metrics development.

        3. **Alternatives Agent**: Expert in strategic thinking, creative problem-solving,
           innovation ideation, and strategic option evaluation. Specializes in design
           thinking, scenario planning, and blue ocean strategy identification.

        4. **Verification Agent**: Expert in validation, feasibility assessment, fact-checking,
           and quality assurance. Specializes in risk assessment, compliance verification,
           and implementation barrier analysis.

        5. **Synthesis Agent**: Expert in multi-perspective integration, comprehensive analysis,
           and executive summary creation. Specializes in strategic alignment, conflict
           resolution, and holistic solution development.

        Agent Configuration:
        - All agents use the configured worker_model_name
        - Loops are set based on loops_per_agent parameter
        - Dynamic temperature is enabled for creative responses
        - Streaming is disabled for complete responses
        - Verbose mode follows class configuration

        Returns:
            Dict[str, Agent]: Dictionary containing all 5 specialized agents with keys:
                - 'research': Research Agent instance
                - 'analysis': Analysis Agent instance
                - 'alternatives': Alternatives Agent instance
                - 'verification': Verification Agent instance
                - 'synthesis': Synthesis Agent instance

        Note:
            This method uses @lru_cache(maxsize=1) to ensure agents are only created once
            per HeavySwarm instance, improving performance for multiple task executions.
        """
        if self.verbose:
            logger.info("🏗️ Creating specialized agents...")

        # Research Agent - Deep information gathering and data collection
        research_agent = Agent(
            agent_name="Research-Agent",
            agent_description="Expert research agent specializing in comprehensive information gathering and data collection",
            system_prompt=RESEARCH_AGENT_PROMPT,
            max_loops=self.loops_per_agent,
            model_name=self.worker_model_name,
            streaming_on=False,
            verbose=False,
            dynamic_temperature_enabled=True,
            print_on=self.agent_prints_on,
        )

        # Analysis Agent - Pattern recognition and deep analytical insights
        analysis_agent = Agent(
            agent_name="Analysis-Agent",
            agent_description="Expert analytical agent specializing in pattern recognition, data analysis, and insight generation",
            system_prompt=ANALYSIS_AGENT_PROMPT,
            max_loops=self.loops_per_agent,
            model_name=self.worker_model_name,
            streaming_on=False,
            verbose=False,
            dynamic_temperature_enabled=True,
            print_on=self.agent_prints_on,
        )

        # Alternatives Agent - Strategic options and creative solutions
        alternatives_agent = Agent(
            agent_name="Alternatives-Agent",
            agent_description="Expert strategic agent specializing in alternative approaches, creative solutions, and option generation",
            system_prompt=ALTERNATIVES_AGENT_PROMPT,
            max_loops=self.loops_per_agent,
            model_name=self.worker_model_name,
            streaming_on=False,
            verbose=False,
            dynamic_temperature_enabled=True,
            print_on=self.agent_prints_on,
        )

        # Verification Agent - Validation, feasibility assessment, and quality assurance
        verification_agent = Agent(
            agent_name="Verification-Agent",
            agent_description="Expert verification agent specializing in validation, feasibility assessment, and quality assurance",
            system_prompt=VERIFICATION_AGENT_PROMPT,
            max_loops=self.loops_per_agent,
            model_name=self.worker_model_name,
            streaming_on=False,
            verbose=False,
            dynamic_temperature_enabled=True,
            print_on=self.agent_prints_on,
        )

        # Synthesis Agent - Integration and comprehensive analysis
        synthesis_agent = Agent(
            agent_name="Synthesis-Agent",
            agent_description="Expert synthesis agent specializing in integration, comprehensive analysis, and final recommendations",
            system_prompt=SYNTHESIS_AGENT_PROMPT,
            max_loops=1,
            model_name=self.worker_model_name,
            streaming_on=False,
            verbose=False,
            dynamic_temperature_enabled=True,
        )

        agents = {
            "research": research_agent,
            "analysis": analysis_agent,
            "alternatives": alternatives_agent,
            "verification": verification_agent,
            "synthesis": synthesis_agent,
        }
        return agents

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
                logger.error(
                    f"❌ Error in {agent_type} Agent: {str(e)} Traceback: {traceback.format_exc()}"
                )
                return agent_type, f"Error: {str(e)}"

        # Prepare agent tasks
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
                    logger.error(
                        f"⏰ Timeout for {agent_type} Agent after {self.timeout}s"
                    )
                    results[agent_type.lower()] = (
                        f"Timeout after {self.timeout} seconds"
                    )
                except Exception as e:
                    logger.error(
                        f"❌ Exception in {agent_type} Agent: {str(e)}"
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
        - Real-time status updates with professional Arasaka-inspired styling
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
        agent_configs = [
            (
                "Agent 1",
                "research",
                "white",
                "Gathering comprehensive research data",
            ),
            (
                "Agent 2",
                "analysis",
                "white",
                "Analyzing patterns and generating insights",
            ),
            (
                "Agent 3",
                "alternatives",
                "white",
                "Exploring creative solutions and alternatives",
            ),
            (
                "Agent 4",
                "verification",
                "white",
                "Validating findings and checking feasibility",
            ),
        ]

        results = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:

            # Create progress tasks for each agent
            tasks = {}
            for (
                display_name,
                agent_key,
                color,
                description,
            ) in agent_configs:
                task_id = progress.add_task(
                    f"[{color}]{display_name}[/{color}]: INITIALIZING",
                    total=None,
                )
                tasks[agent_key] = task_id

            # Define agent execution function with progress updates
            def execute_agent_with_progress(agent_info):
                agent_type, agent_key, agent, question = agent_info
                try:
                    # Update progress to show agent starting
                    progress.update(
                        tasks[agent_key],
                        description=f"[red]{agent_type}[/red]: INITIALIZING ••••••••",
                    )

                    # Simulate some processing time for visual effect
                    time.sleep(0.5)
                    progress.update(
                        tasks[agent_key],
                        description=f"[red]{agent_type}[/red]: PROCESSING QUERY ••••••••••••••",
                    )

                    # Execute the agent with dots animation
                    progress.update(
                        tasks[agent_key],
                        description=f"[red]{agent_type}[/red]: EXECUTING ••••••••••••••••••••",
                    )

                    result = agent.run(question)

                    # Update progress during execution
                    progress.update(
                        tasks[agent_key],
                        description=f"[white]{agent_type}[/white]: GENERATING RESPONSE ••••••••••••••••••••••••••",
                    )

                    # Add to conversation
                    self.conversation.add(
                        role=agent.agent_name,
                        content=result,
                        category="output",
                    )

                    # Complete the progress
                    progress.update(
                        tasks[agent_key],
                        description=f"[bold white]{agent_type}[/bold white]: ✅ COMPLETE! ••••••••••••••••••••••••••••••••",
                    )

                    return agent_type, result

                except Exception as e:
                    progress.update(
                        tasks[agent_key],
                        description=f"[bold red]{agent_type}[/bold red]: ❌ ERROR! ••••••••••••••••••••••••••••••••",
                    )
                    logger.error(
                        f"❌ Error in {agent_type} Agent: {str(e)} Traceback: {traceback.format_exc()}"
                    )
                    return agent_type, f"Error: {str(e)}"

            # Prepare agent tasks with keys
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
                    questions.get("alternatives_question", ""),
                ),
                (
                    "Agent 4",
                    "verification",
                    agents["verification"],
                    questions.get("verification_question", ""),
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
                        progress.update(
                            tasks[agent_key],
                            description=f"[bold red]Agent {list(tasks.keys()).index(agent_key) + 1}[/bold red]: ⏰ TIMEOUT! ••••••••••••••••••••••••••••••••",
                        )
                        results[agent_key] = (
                            f"Timeout after {self.timeout} seconds"
                        )
                    except Exception as e:
                        progress.update(
                            tasks[agent_key],
                            description=f"[bold red]Agent {list(tasks.keys()).index(agent_key) + 1}[/bold red]: ❌ ERROR! ••••••••••••••••••••••••••••••••",
                        )
                        results[agent_key] = f"Exception: {str(e)}"

        # Show completion summary
        self.console.print(
            Panel(
                "[bold red]⚡ ALL AGENTS COMPLETED SUCCESSFULLY![/bold red]\n"
                "[white]Results from all 4 specialized agents are ready for synthesis[/white]",
                title="[bold red]EXECUTION COMPLETE[/bold red]",
                border_style="red",
            )
        )
        self.console.print()

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
        # Get the cached agents
        agents = self.create_agents()
        synthesis_agent = agents["synthesis"]

        agents_names = [
            "Research Agent",
            "Analysis Agent",
            "Alternatives Agent",
            "Verification Agent",
        ]

        # Create comprehensive synthesis prompt
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

            return {
                "thinking": arguments.get("thinking", ""),
                "research_question": arguments.get(
                    "research_question", ""
                ),
                "analysis_question": arguments.get(
                    "analysis_question", ""
                ),
                "alternatives_question": arguments.get(
                    "alternatives_question", ""
                ),
                "verification_question": arguments.get(
                    "verification_question", ""
                ),
                "tool_call_id": tool_call.id,
                "function_name": tool_call.function.name,
            }

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
        prompt = f"""
        You are an expert task analyzer. Your job is to break down the following task into 4 specialized questions for different agent roles:

        1. Research Agent: Focuses on gathering information, data, and background context
        2. Analysis Agent: Focuses on examining patterns, trends, and deriving insights  
        3. Alternatives Agent: Focuses on exploring different approaches and solutions
        4. Verification Agent: Focuses on validating findings and checking feasibility

        Task to analyze: {task}

        Use the generate_specialized_questions function to create targeted questions for each agent role.
        """

        question_agent = LiteLLM(
            system_prompt=prompt,
            model=self.question_agent_model_name,
            tools_list_dictionary=schema,
            max_tokens=3000,
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

        if self.verbose:
            logger.info(
                f"🔍 Question Generation Output: {out} and type: {type(out)}"
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

        return [
            questions.get("research_question", ""),
            questions.get("analysis_question", ""),
            questions.get("alternatives_question", ""),
            questions.get("verification_question", ""),
        ]
