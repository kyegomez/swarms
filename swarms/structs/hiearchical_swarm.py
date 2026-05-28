"""
Hierarchical Swarm Implementation

This module provides a hierarchical swarm architecture where a director agent coordinates
multiple worker agents to execute complex tasks through a structured workflow.

Flow:
1. User provides a task
2. Director creates a plan
3. Director distributes orders to agents individually or multiple tasks at once
4. Agents execute tasks and report back to the director
5. Director evaluates results and issues new orders if needed (up to max_loops)
6. All context and conversation history is preserved throughout the process

Todo
- Add layers of management -- a list of list of agents that act as departments
- Auto build agents from input prompt - and then add them to the swarm
- Make it faster and more high performance
- Enable the director to choose a multi-agent approach to the task, it orchestrates how the agents talk and work together.
- Improve the director feedback, maybe add agent as a judge to the worker agent instead of the director.
"""

import asyncio
import json
import os
import queue as _queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from swarms.prompts.agent_judge_prompt import (
    HIERARCHICAL_SWARM_JUDGE_PROMPT,
)
from swarms.prompts.hiearchical_system_prompt import (
    DIRECTOR_PLANNING_PROMPT,
    HIEARCHICAL_SWARM_SYSTEM_PROMPT,
)
from swarms.prompts.multi_agent_collab_prompt import (
    MULTI_AGENT_COLLAB_PROMPT_TWO,
)
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.ma_utils import list_all_agents
from swarms.structs.omni_agent_types import AgentListType
from swarms.tools.base_tool import BaseTool
from swarms.utils.formatter import formatter
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.output_types import OutputType
from swarms.utils.swarm_autosave import get_swarm_workspace_dir
from swarms.utils.workspace_utils import get_workspace_dir


class HierarchicalSwarmDashboard:
    """
    Futuristic Swarms Corporation-style dashboard for hierarchical swarm monitoring.

    This dashboard provides a professional, enterprise-grade interface with red and black
    color scheme, real-time monitoring of swarm operations, and cyberpunk aesthetics.

    Attributes:
        console (Console): Rich console instance for rendering
        live_display (Live): Live display for real-time updates
        swarm_name (str): Name of the swarm being monitored
        agent_statuses (dict): Current status of all agents
        director_status (str): Current status of the director
        current_loop (int): Current execution loop
        max_loops (int): Maximum number of loops
        is_active (bool): Whether the dashboard is currently active
    """

    def __init__(self, swarm_name: str = "Swarms Corporation"):
        """
        Initialize the Hierarchical Swarms dashboard.

        Args:
            swarm_name (str): Name of the swarm to display in the dashboard
        """
        self.console = Console()
        self.live_display = None
        self.swarm_name = swarm_name
        self.agent_statuses = {}
        self.director_status = "INITIALIZING"
        self.current_loop = 0
        self.max_loops = 1
        self.is_active = False
        self.start_time = None
        self.spinner_frames = [
            "⠋",
            "⠙",
            "⠹",
            "⠸",
            "⠼",
            "⠴",
            "⠦",
            "⠧",
            "⠇",
            "⠏",
        ]
        self.spinner_idx = 0

        # Director information tracking
        self.director_plan = ""
        self.director_orders = []

        # Swarm information
        self.swarm_description = ""
        self.director_name = "Director"
        self.director_model_name = "gpt-5.4"

        # View mode for agents display
        self.detailed_view = False

        # Multi-loop agent tracking
        self.agent_history = {}  # Track agent outputs across loops
        self.current_loop = 0

    def _get_spinner(self) -> str:
        """Get current spinner frame for loading animations."""
        self.spinner_idx = (self.spinner_idx + 1) % len(
            self.spinner_frames
        )
        return self.spinner_frames[self.spinner_idx]

    def _create_header(self) -> Panel:
        """Create the dashboard header with Swarms Corporation branding."""
        header_text = Text()
        header_text.append(
            "╔══════════════════════════════════════════════════════════════════════════════╗\n",
            style="bold red",
        )
        header_text.append("║", style="bold red")
        header_text.append("                    ", style="bold red")
        header_text.append(
            "SWARMS CORPORATION", style="bold white on red"
        )
        header_text.append("                    ", style="bold red")
        header_text.append("║\n", style="bold red")
        header_text.append("║", style="bold red")
        header_text.append("                    ", style="bold red")
        header_text.append(
            "HIERARCHICAL SWARM OPERATIONS CENTER", style="bold red"
        )
        header_text.append("                    ", style="bold red")
        header_text.append("║\n", style="bold red")
        header_text.append(
            "╚══════════════════════════════════════════════════════════════════════════════╝",
            style="bold red",
        )

        return Panel(
            header_text,
            border_style="red",
            padding=(0, 1),
        )

    def _create_status_panel(self) -> Panel:
        """Create the operations status panel."""
        status_text = Text()

        # Corporation branding and operation type
        status_text.append(
            "By the Swarms Corporation", style="bold cyan"
        )
        status_text.append("\n", style="white")
        status_text.append(
            "Hierarchical Agent Operations", style="bold white"
        )

        status_text.append("\n\n", style="white")

        # Swarm information
        status_text.append("SWARM NAME: ", style="bold white")
        status_text.append(f"{self.swarm_name}", style="bold cyan")

        status_text.append("\n", style="white")
        status_text.append("DESCRIPTION: ", style="bold white")
        status_text.append(f"{self.swarm_description}", style="white")

        status_text.append("\n", style="white")
        status_text.append("DIRECTOR: ", style="bold white")
        status_text.append(
            f"{self.director_name} ({self.director_model_name})",
            style="cyan",
        )

        status_text.append("\n", style="white")
        status_text.append("TOTAL LOOPS: ", style="bold white")
        status_text.append(f"{self.max_loops}", style="bold cyan")

        status_text.append("  |  ", style="white")
        status_text.append("CURRENT LOOP: ", style="bold white")
        status_text.append(
            f"{self.current_loop}", style="bold yellow"
        )

        # Agent count metadata
        agent_count = len(getattr(self, "agent_history", {}))
        status_text.append("  |  ", style="white")
        status_text.append("AGENTS: ", style="bold white")
        status_text.append(f"{agent_count}", style="bold green")

        status_text.append("\n\n", style="white")

        # Director status
        status_text.append("DIRECTOR STATUS: ", style="bold white")
        if self.director_status == "INITIALIZING":
            status_text.append(
                f"{self._get_spinner()} {self.director_status}",
                style="bold yellow",
            )
        elif self.director_status == "ACTIVE":
            status_text.append(
                f"✓ {self.director_status}", style="bold green"
            )
        elif self.director_status == "PROCESSING":
            status_text.append(
                f"{self._get_spinner()} {self.director_status}",
                style="bold cyan",
            )
        else:
            status_text.append(
                f"✗ {self.director_status}", style="bold red"
            )

        status_text.append("\n\n", style="white")

        # Runtime and completion information
        if self.start_time:
            runtime = time.time() - self.start_time
            status_text.append("RUNTIME: ", style="bold white")
            status_text.append(f"{runtime:.2f}s", style="bold green")

            # Add completion percentage if loops are running
            if self.max_loops > 0:
                completion_percent = (
                    self.current_loop / self.max_loops
                ) * 100
                status_text.append("  |  ", style="white")
                status_text.append("PROGRESS: ", style="bold white")
                status_text.append(
                    f"{completion_percent:.1f}%", style="bold cyan"
                )

        return Panel(
            status_text,
            border_style="red",
            padding=(1, 2),
            title="[bold white]OPERATIONS STATUS[/bold white]",
        )

    def _create_agents_table(self) -> Table:
        """Create the agents monitoring table with full outputs and loop history."""
        table = Table(
            show_header=True,
            header_style="bold white on red",
            border_style="red",
            title="[bold white]AGENT MONITORING MATRIX[/bold white]",
            title_style="bold white",
            show_lines=True,
        )

        table.add_column("AGENT ID", style="bold cyan", width=25)
        table.add_column("LOOP", style="bold white", width=8)
        table.add_column("STATUS", style="bold white", width=15)
        table.add_column("TASK", style="white", width=40)
        table.add_column("OUTPUT", style="white", width=150)

        # Display agents with their history across loops
        for agent_name, history in self.agent_history.items():
            for loop_num in range(self.max_loops + 1):
                loop_key = f"Loop_{loop_num}"

                if loop_key in history:
                    loop_data = history[loop_key]
                    status = loop_data.get("status", "UNKNOWN")
                    task = loop_data.get("task", "N/A")
                    output = loop_data.get("output", "")

                    # Style status
                    if status == "RUNNING":
                        status_display = (
                            f"{self._get_spinner()} {status}"
                        )
                        status_style = "bold yellow"
                    elif status == "COMPLETED":
                        status_display = f"✓ {status}"
                        status_style = "bold green"
                    elif status == "PENDING":
                        status_display = f"○ {status}"
                        status_style = "bold red"
                    else:
                        status_display = f"✗ {status}"
                        status_style = "bold red"

                    # Show full output without truncation
                    output_display = output if output else "No output"

                    table.add_row(
                        Text(agent_name, style="bold cyan"),
                        Text(f"Loop {loop_num}", style="bold white"),
                        Text(status_display, style=status_style),
                        Text(task, style="white"),
                        Text(output_display, style="white"),
                    )

        return table

    def _create_detailed_agents_view(self) -> Panel:
        """Create a detailed view of agents with full outputs and loop history."""
        detailed_text = Text()

        for agent_name, history in self.agent_history.items():
            detailed_text.append(
                f"AGENT: {agent_name}\n", style="bold cyan"
            )
            detailed_text.append("=" * 80 + "\n", style="red")

            for loop_num in range(self.max_loops + 1):
                loop_key = f"Loop_{loop_num}"

                if loop_key in history:
                    loop_data = history[loop_key]
                    status = loop_data.get("status", "UNKNOWN")
                    task = loop_data.get("task", "N/A")
                    output = loop_data.get("output", "")

                    detailed_text.append(
                        f"LOOP {loop_num}:\n", style="bold white"
                    )
                    detailed_text.append(
                        f"STATUS: {status}\n", style="bold white"
                    )
                    detailed_text.append(
                        f"TASK: {task}\n", style="white"
                    )
                    detailed_text.append(
                        "OUTPUT:\n", style="bold white"
                    )
                    detailed_text.append(f"{output}\n", style="white")
                    detailed_text.append("─" * 80 + "\n", style="red")

        return Panel(
            detailed_text,
            border_style="red",
            padding=(1, 2),
            title="[bold white]DETAILED AGENT OUTPUTS (FULL HISTORY)[/bold white]",
        )

    def _create_director_panel(self) -> Panel:
        """Create the director information panel showing plan and orders."""
        director_text = Text()

        # Plan section
        director_text.append("DIRECTOR PLAN:\n", style="bold white")
        if self.director_plan:
            director_text.append(self.director_plan, style="white")
        else:
            director_text.append(
                "No plan available", style="dim white"
            )

        director_text.append("\n\n", style="white")

        # Orders section
        director_text.append("CURRENT ORDERS:\n", style="bold white")
        if self.director_orders:
            for i, order in enumerate(
                self.director_orders
            ):  # Show first 5 orders
                director_text.append(f"{i+1}. ", style="bold cyan")
                director_text.append(
                    f"{order.get('agent_name', 'Unknown')}: ",
                    style="bold white",
                )
                task = order.get("task", "No task")
                director_text.append(task, style="white")
                director_text.append("\n", style="white")

            if len(self.director_orders) > 5:
                director_text.append(
                    f"... and {len(self.director_orders) - 5} more orders",
                    style="dim white",
                )
        else:
            director_text.append(
                "No orders available", style="dim white"
            )

        return Panel(
            director_text,
            border_style="red",
            padding=(1, 2),
            title="[bold white]DIRECTOR OPERATIONS[/bold white]",
        )

    def _create_dashboard_layout(self) -> Layout:
        """Create the complete dashboard layout."""
        layout = Layout()

        # Split into operations status, director operations, and agents
        layout.split_column(
            Layout(name="operations_status", size=12),
            Layout(name="director_operations", size=12),
            Layout(name="agents", ratio=1),
        )

        # Add content to each section
        layout["operations_status"].update(
            self._create_status_panel()
        )
        layout["director_operations"].update(
            self._create_director_panel()
        )

        # Choose between table view and detailed view
        if self.detailed_view:
            layout["agents"].update(
                self._create_detailed_agents_view()
            )
        else:
            layout["agents"].update(
                Panel(
                    self._create_agents_table(),
                    border_style="red",
                    padding=(1, 1),
                )
            )

        return layout

    def start(self, max_loops: int = 1):
        """Start the dashboard display."""
        self.max_loops = max_loops
        self.start_time = time.time()
        self.is_active = True

        self.live_display = Live(
            self._create_dashboard_layout(),
            console=self.console,
            refresh_per_second=10,
            transient=False,
        )
        self.live_display.start()

    def update_agent_status(
        self,
        agent_name: str,
        status: str,
        task: str = "",
        output: str = "",
    ):
        """Update the status of a specific agent."""
        # Create loop key for tracking history
        loop_key = f"Loop_{self.current_loop}"

        # Initialize agent history if not exists
        if agent_name not in self.agent_history:
            self.agent_history[agent_name] = {}

        # Store current status and add to history
        self.agent_statuses[agent_name] = {
            "status": status,
            "task": task,
            "output": output,
        }

        # Add to history for this loop
        self.agent_history[agent_name][loop_key] = {
            "status": status,
            "task": task,
            "output": output,
        }

        if self.live_display and self.is_active:
            self.live_display.update(self._create_dashboard_layout())

    def update_director_status(self, status: str):
        """Update the director status."""
        self.director_status = status
        if self.live_display and self.is_active:
            self.live_display.update(self._create_dashboard_layout())

    def update_loop(self, current_loop: int):
        """Update the current execution loop."""
        self.current_loop = current_loop
        if self.live_display and self.is_active:
            self.live_display.update(self._create_dashboard_layout())

    def update_director_plan(self, plan: str):
        """Update the director's plan."""
        self.director_plan = plan
        if self.live_display and self.is_active:
            self.live_display.update(self._create_dashboard_layout())

    def update_director_orders(self, orders: list):
        """Update the director's orders."""
        self.director_orders = orders
        if self.live_display and self.is_active:
            self.live_display.update(self._create_dashboard_layout())

    def stop(self):
        """Stop the dashboard display."""
        self.is_active = False
        if self.live_display:
            self.live_display.stop()
            self.console.print()

    def update_swarm_info(
        self,
        name: str,
        description: str,
        max_loops: int,
        director_name: str,
        director_model_name: str,
    ):
        """Update the dashboard with swarm-specific information."""
        self.swarm_name = name
        self.swarm_description = description
        self.max_loops = max_loops
        self.director_name = director_name
        self.director_model_name = director_model_name
        if self.live_display and self.is_active:
            self.live_display.update(self._create_dashboard_layout())

    def force_refresh(self):
        """Force refresh the dashboard display."""
        if self.live_display and self.is_active:
            self.live_display.update(self._create_dashboard_layout())

    def show_full_output(self, agent_name: str, full_output: str):
        """Display full agent output in a separate panel."""
        if self.live_display and self.is_active:
            # Create a full output panel
            output_panel = Panel(
                Text(full_output, style="white"),
                title=f"[bold white]FULL OUTPUT - {agent_name}[/bold white]",
                border_style="red",
                padding=(1, 2),
                width=120,
            )

            # Temporarily show the full output
            self.console.print(output_panel)
            self.console.print()  # Add spacing

    def toggle_detailed_view(self):
        """Toggle between table view and detailed view."""
        self.detailed_view = not self.detailed_view
        if self.live_display and self.is_active:
            self.live_display.update(self._create_dashboard_layout())


class HierarchicalOrder(BaseModel):
    """
    Represents a single task assignment within the hierarchical swarm.

    This class defines the structure for individual task orders that the director
    distributes to worker agents. Each order specifies which agent should execute
    what specific task.

    Attributes:
        agent_name (str): The name of the agent assigned to execute the task.
                         Must match an existing agent in the swarm.
        task (str): The specific task description to be executed by the assigned agent.
                   Should be clear and actionable.
    """

    agent_name: str = Field(
        ...,
        description="Specifies the name of the agent to which the task is assigned. This is a crucial element in the hierarchical structure of the swarm, as it determines the specific agent responsible for the task execution.",
    )
    task: str = Field(
        ...,
        description="Defines the specific task to be executed by the assigned agent. This task is a key component of the swarm's plan and is essential for achieving the swarm's goals.",
    )


class HierarchicalOrderRearrange(BaseModel):
    """
    Represents a single task assignment within the hierarchical swarm.

    This class defines the structure for individual task orders that the director
    distributes to worker agents. Each order specifies which agent should execute
    what specific task.
    """

    initial_task: str = Field(
        ...,
        description="The initial task that the director has to execute.",
    )
    flow_of_communication: str = Field(
        ...,
        description="How the agents will communicate with each other to accomplish the task. Like agent_one -> agent_two -> agent_three -> agent_four -> agent_one, can use comma signs to denote sequential communication and commas to denote parallel communication for example agent_one -> agent_two, agent_three -> agent_four",
    )


class SwarmSpec(BaseModel):
    """
    Defines the complete specification for a hierarchical swarm execution.

    This class contains the overall plan and all individual orders that the director
    creates to coordinate the swarm's activities. It serves as the structured output
    format for the director agent.

    Attributes:
        plan (str): A comprehensive plan outlining the sequence of actions and strategy
                   for the entire swarm to accomplish the given task.
        orders (List[HierarchicalOrder]): A list of specific task assignments to
                                         individual agents within the swarm.
    """

    plan: str = Field(
        ...,
        description="A plan generated by the director agent for the swarm to accomplish the given task, where the director autonomously reasons through the problem, devises its own strategy, and determines the sequence of actions. "
        "This plan reflects the director's independent thought process, outlining the rationale, priorities, and steps it deems necessary for successful execution. "
        "It serves as a blueprint for the swarm, enabling agents to follow the director's self-derived guidance and adapt as needed throughout the process.",
    )

    orders: List[HierarchicalOrder] = Field(
        ...,
        description="A collection of task assignments to specific agents within the swarm. These orders are the specific instructions that guide the agents in their task execution and are a key element in the swarm's plan.",
    )


class AgentScore(BaseModel):
    agent_name: str
    score: int = Field(..., ge=0, le=10)
    reasoning: str
    suggestions: str


class JudgeReport(BaseModel):
    overall_quality: int = Field(..., ge=0, le=10)
    scores: List[AgentScore]
    summary: str


class HierarchicalSwarm:
    """
    A hierarchical swarm orchestrator that coordinates multiple agents through a director.

    This class implements a hierarchical architecture where a director agent creates
    plans and distributes tasks to worker agents. The director can provide feedback
    and iterate on results through multiple loops to achieve the desired outcome.

    The swarm maintains conversation history throughout the process, allowing for
    context-aware decision making and iterative refinement of results.

    Attributes:
        name (str): The name identifier for this swarm instance.
        description (str): A description of the swarm's purpose and capabilities.
        director (Optional[Union[Agent, Callable, Any]]): The director agent that
                                                         coordinates the swarm.
        agents (List[Union[Agent, Callable, Any]]): List of worker agents available
                                                   for task execution.
        max_loops (int): Maximum number of feedback loops the swarm can perform.
        output_type (OutputType): Format for the final output of the swarm.
        feedback_director_model_name (str): Model name for the feedback director.
        director_name (str): Name identifier for the director agent.
        director_model_name (str): Model name for the main director agent.
        add_collaboration_prompt (bool): Whether to add collaboration prompts to agents.
        director_feedback_on (bool): Whether director feedback is enabled.
        parallel_execution (bool): Whether to execute agent tasks in parallel (default: True).
    """

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
        director_feedback_on: bool = True,
        interactive: bool = False,
        director_system_prompt: str = HIEARCHICAL_SWARM_SYSTEM_PROMPT,
        multi_agent_prompt_improvements: bool = False,
        director_temperature: float = 0.7,
        director_top_p: float = 0.9,
        planning_enabled: bool = True,
        autosave: bool = True,
        verbose: bool = False,
        parallel_execution: bool = True,
        agent_as_judge: bool = False,
        judge_agent_model_name: str = "gpt-5.4",
        *args,
        **kwargs,
    ):
        """
        Initialize a new HierarchicalSwarm instance.

        Args:
            name (str): The name identifier for this swarm instance.
            description (str): A description of the swarm's purpose.
            director (Optional[Union[Agent, Callable, Any]]): The director agent.
                                                             If None, a default director will be created.
            agents (List[Union[Agent, Callable, Any]]): List of worker agents.
                                                       Must not be empty.
            max_loops (int): Maximum number of feedback loops (must be > 0).
            output_type (OutputType): Format for the final output.
            feedback_director_model_name (str): Model name for feedback director.
            director_name (str): Name identifier for the director agent.
            director_model_name (str): Model name for the main director agent.
            add_collaboration_prompt (bool): Whether to add collaboration prompts.
            director_feedback_on (bool): Whether director feedback is enabled.
            autosave (bool): Whether to enable autosaving of conversation history.
            verbose (bool): Whether to enable verbose logging.
            parallel_execution (bool): Whether to execute agent tasks in parallel (default: True).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If no agents are provided or max_loops is invalid.
        """
        self.name = name
        self.description = description
        self.director = director
        self.agents = agents
        # O(1) agent lookup via dict - keyed by agent_name
        self.agent_map = {
            agent.agent_name: agent
            for agent in (agents or [])
            if hasattr(agent, "agent_name")
        }
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
        self.planning_enabled = planning_enabled
        self.autosave = autosave
        self.verbose = verbose
        self.parallel_execution = parallel_execution
        self.agent_as_judge = agent_as_judge
        self.judge_agent_model_name = judge_agent_model_name
        self.swarm_workspace_dir = None

        # Setup autosave workspace if enabled
        if self.autosave:
            self._setup_autosave()

        self.initialize_swarm()

    def initialize_swarm(self):
        if self.interactive:
            self.agents_no_print()

        # Initialize dashboard if interactive mode is enabled
        self.dashboard = None
        if self.interactive:
            self.dashboard = HierarchicalSwarmDashboard(self.name)
            # Enable detailed view for better output visibility
            self.dashboard.detailed_view = True
            # Pass additional swarm information to dashboard
            self.dashboard.update_swarm_info(
                name=self.name,
                description=self.description,
                max_loops=self.max_loops,
                director_name=self.director_name,
                director_model_name=self.director_model_name,
            )

        self.init_swarm()

    def list_worker_agents(self) -> str:
        return list_all_agents(
            agents=self.agents,
            add_to_conversation=False,
        )

    def display_hierarchy(self) -> None:
        """
        Display the hierarchical structure of the swarm using Rich Tree.

        This method creates a visual tree representation showing the Director
        at the top level and all worker agents as children branches. The tree
        is printed to the console with rich formatting.

        The hierarchy visualization helps understand the organizational structure
        of the swarm, with the Director coordinating all worker agents.
        """
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
        """
        Initialize the swarm with proper configuration and validation.

        This method performs the following initialization steps:
        1. Sets up logging if verbose mode is enabled
        2. Creates a conversation instance for history tracking
        3. Performs reliability checks on the configuration
        4. Adds agent context to the director

        Raises:
            ValueError: If the swarm configuration is invalid.
        """
        self.conversation = Conversation(time_enabled=False)

        # Reliability checks
        self.reliability_checks()

        # Add agent context to the director
        self.add_context_to_director()

        # Initialize agent statuses in dashboard if interactive mode
        if self.interactive and self.dashboard:
            for agent in self.agents:
                if hasattr(agent, "agent_name"):
                    self.dashboard.update_agent_status(
                        agent.agent_name,
                        "PENDING",
                        "Awaiting task assignment",
                        "Ready for deployment",
                    )
            # Force refresh to ensure agents are displayed
            self.dashboard.force_refresh()

        if self.multi_agent_prompt_improvements:
            self.prepare_worker_agents()

    def _setup_autosave(self):
        """
        Setup workspace directory for saving conversation history.

        Creates the workspace directory structure if autosave is enabled.
        Only conversation history will be saved to this directory.
        """
        try:
            # Set default workspace directory if not set
            if not os.getenv("WORKSPACE_DIR"):
                default_workspace = os.path.join(
                    os.getcwd(), "agent_workspace"
                )
                os.environ["WORKSPACE_DIR"] = default_workspace
                # Clear the cache so get_workspace_dir() picks up the new value
                get_workspace_dir.cache_clear()
                if self.verbose:
                    logger.info(
                        f"WORKSPACE_DIR not set, using default: {default_workspace}"
                    )

            class_name = self.__class__.__name__
            swarm_name = self.name or "hierarchical-swarm"
            self.swarm_workspace_dir = get_swarm_workspace_dir(
                class_name, swarm_name, use_timestamp=True
            )

            if self.swarm_workspace_dir:
                if self.verbose:
                    logger.info(
                        f"Autosave enabled. Conversation history will be saved to: {self.swarm_workspace_dir}"
                    )
        except Exception as e:
            logger.warning(
                f"Failed to setup autosave for HierarchicalSwarm: {e}"
            )
            # Don't raise - autosave failures shouldn't break initialization
            self.swarm_workspace_dir = None

    def _save_conversation_history(self):
        """
        Save conversation history as a separate JSON file to the workspace directory.

        Saves the conversation history to:
        workspace_dir/swarms/HierarchicalSwarm/{swarm-name}-{id}/conversation_history.json
        """
        if not self.swarm_workspace_dir:
            return

        try:
            # Get conversation history
            if hasattr(self, "conversation") and self.conversation:
                if hasattr(self.conversation, "conversation_history"):
                    conversation_data = (
                        self.conversation.conversation_history
                    )
                elif hasattr(self.conversation, "to_dict"):
                    conversation_data = self.conversation.to_dict()
                else:
                    conversation_data = []

                # Create conversation history file path
                conversation_path = os.path.join(
                    self.swarm_workspace_dir,
                    "conversation_history.json",
                )

                # Save conversation history as JSON
                with open(
                    conversation_path, "w", encoding="utf-8"
                ) as f:
                    json.dump(
                        conversation_data,
                        f,
                        indent=2,
                        default=str,
                    )

                if self.verbose:
                    logger.debug(
                        f"Saved conversation history to {conversation_path}"
                    )
            else:
                if self.verbose:
                    logger.debug(
                        "No conversation object found, skipping conversation history save"
                    )
        except Exception as e:
            logger.warning(
                f"Failed to save conversation history: {e}"
            )

    def add_context_to_director(self):
        """
        Add agent context and collaboration information to the director's conversation.

        This method ensures that the director has complete information about all
        available agents, their capabilities, and how they can collaborate. This
        context is essential for the director to make informed decisions about
        task distribution.

        Raises:
            Exception: If adding context fails due to agent configuration issues.
        """
        try:
            list_all_agents(
                agents=self.agents,
                conversation=self.conversation,
                add_to_conversation=True,
                add_collaboration_prompt=self.add_collaboration_prompt,
            )

        except Exception as e:
            error_msg = (
                f"[ERROR] Failed to add context to director: {str(e)}"
            )
            logger.error(
                f"{error_msg}\n[TRACE] Traceback: {traceback.format_exc()}"
            )

    def setup_director(self):
        """
        Set up the director agent with proper configuration and tools.

        Creates a new director agent with the SwarmSpec schema for structured
        output, enabling it to create plans and distribute orders effectively.

        Returns:
            Agent: A configured director agent ready to coordinate the swarm.

        Raises:
            Exception: If director setup fails due to configuration issues.
        """
        try:
            schema = BaseTool().base_model_to_dict(SwarmSpec)

            return Agent(
                agent_name=self.director_name,
                agent_description="A director agent that can create a plan and distribute orders to agents",
                system_prompt=self.director_system_prompt,
                model_name=self.director_model_name,
                temperature=self.director_temperature,
                top_p=self.director_top_p,
                max_loops=1,
                base_model=SwarmSpec,
                tools_list_dictionary=[schema],
                output_type="dict-all-except-first",
            )

        except Exception as e:
            error_msg = f"[ERROR] Failed to setup director: {str(e)}"
            logger.error(
                f"{error_msg}\n[TRACE] Traceback: {traceback.format_exc()}\n[BUG] If this issue persists, please report it at: https://github.com/kyegomez/swarms/issues"
            )

    def setup_director_with_planning(
        self, task: str = None, img: Optional[str] = None
    ):
        try:

            agent = Agent(
                agent_name=self.director_name,
                agent_description="A director agent that can create a plan and distribute orders to agents",
                system_prompt=DIRECTOR_PLANNING_PROMPT,
                model_name=self.director_model_name,
                temperature=self.director_temperature,
                top_p=self.director_top_p,
                max_loops=1,
                output_type="final",
            )

            return agent.run(task=task, img=img)

        except Exception as e:
            error_msg = f"[ERROR] Failed to setup director with planning: {str(e)}"
            logger.error(
                f"{error_msg}\n[TRACE] Traceback: {traceback.format_exc()}\n[BUG] If this issue persists, please report it at: https://github.com/kyegomez/swarms/issues"
            )

    def reliability_checks(self):
        """
        Perform validation checks to ensure the swarm is properly configured.

        This method validates:
        1. That at least one agent is provided
        2. That max_loops is greater than 0
        3. That a director is available (creates default if needed)

        Raises:
            ValueError: If the swarm configuration is invalid.
        """
        try:
            if not self.agents or len(self.agents) == 0:
                raise ValueError(
                    "No agents found in the swarm. At least one agent must be provided to create a hierarchical swarm."
                )

            if self.max_loops <= 0:
                raise ValueError(
                    "Max loops must be greater than 0. Please set a valid number of loops."
                )

            if self.director is None:
                self.director = self.setup_director()

        except Exception as e:
            error_msg = f"[ERROR] Reliability checks failed: {str(e)}"
            logger.error(
                f"{error_msg}\n[TRACE] Traceback: {traceback.format_exc()}\n[BUG] If this issue persists, please report it at: https://github.com/kyegomez/swarms/issues"
            )
            raise e

    def agents_no_print(self):
        for agent in self.agents:
            agent.print_on = False

    def run_director(
        self,
        task: str,
        img: str = None,
    ) -> SwarmSpec:
        """
        Execute the director agent with the given task and conversation context.

        This method runs the director agent to create a plan and distribute orders
        based on the current task and conversation history.

        Args:
            task (str): The task to be executed by the director.
            img (str, optional): Optional image input for the task.

        Returns:
            SwarmSpec: The director's output containing the plan and orders.

        Raises:
            Exception: If director execution fails.
        """
        try:
            if self.planning_enabled is True:
                self.director.tools_list_dictionary = None
                out = self.setup_director_with_planning(
                    task=f"History: {self.conversation.get_str()} \n\n Task: {task}",
                    img=img,
                )
                self.conversation.add(
                    role=self.director.agent_name, content=out
                )

            # Run the director with the context
            function_call = self.director.run(
                task=f"History: {self.conversation.get_str()} \n\n Task: {task}",
                img=img,
            )

            self.conversation.add(
                role="Director", content=function_call
            )

            return function_call

        except Exception as e:
            error_msg = f"[ERROR] Failed to run director: {str(e)}"
            logger.error(
                f"{error_msg}\n[TRACE] Traceback: {traceback.format_exc()}\n[BUG] If this issue persists, please report it at: https://github.com/kyegomez/swarms/issues"
            )
            raise e

    def step(
        self,
        task: str,
        img: str = None,
        streaming_callback: Optional[
            Callable[[str, str, bool], None]
        ] = None,
        *args,
        **kwargs,
    ):
        """
        Execute a single step of the hierarchical swarm workflow.

        This method performs one complete iteration of the swarm's workflow:
        1. Run the director to create a plan and orders
        2. Parse the director's output to extract plan and orders
        3. Execute all orders by calling the appropriate agents
        4. Optionally generate director feedback on the results

        Args:
            task (str): The task to be processed in this step.
            img (str, optional): Optional image input for the task.
            streaming_callback (Callable[[str, str, bool], None], optional):
                Callback function for streaming agent outputs. Parameters are
                (agent_name, chunk, is_final) where is_final indicates completion.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The results from this step, either agent outputs or director feedback.

        Raises:
            Exception: If step execution fails.
        """
        try:
            # Update dashboard for director execution
            if self.interactive and self.dashboard:
                self.dashboard.update_director_status("PLANNING")

            output = self.run_director(task=task, img=img)

            # Parse the orders
            plan, orders = self.parse_orders(output)

            formatter.print_director_task_distribution(
                director_name=self.director_name,
                orders=orders,
            )

            # Update dashboard with plan and orders information
            if self.interactive and self.dashboard:
                self.dashboard.update_director_plan(plan)
                # Convert orders to list of dicts for dashboard
                orders_list = [
                    {
                        "agent_name": order.agent_name,
                        "task": order.task,
                    }
                    for order in orders
                ]
                self.dashboard.update_director_orders(orders_list)
                self.dashboard.update_director_status("EXECUTING")

            # Execute the orders
            outputs = self.execute_orders(
                orders, streaming_callback=streaming_callback
            )

            if self.agent_as_judge:
                feedback = self.run_judge_agent(outputs)
            elif self.director_feedback_on is True:
                feedback = self.feedback_director(outputs)
            else:
                feedback = outputs

            return feedback

        except Exception as e:
            error_msg = f"[ERROR] Step execution failed: {str(e)}"
            logger.error(
                f"{error_msg}\n[TRACE] Traceback: {traceback.format_exc()}\n[BUG] If this issue persists, please report it at: https://github.com/kyegomez/swarms/issues"
            )

    def run(
        self,
        task: Optional[str] = None,
        img: Optional[str] = None,
        streaming_callback: Optional[
            Callable[[str, str, bool], None]
        ] = None,
        *args,
        **kwargs,
    ):
        """
        Execute the hierarchical swarm for the specified number of feedback loops.

        This method orchestrates the complete swarm execution, performing multiple
        iterations based on the max_loops configuration. Each iteration builds upon
        the previous results, allowing for iterative refinement and improvement.

        The method maintains conversation history throughout all loops and provides
        context from previous iterations to subsequent ones.

        Args:
            task (str, optional): The initial task to be processed by the swarm.
                                 If None and interactive mode is enabled, will prompt for input.
            img (str, optional): Optional image input for the agents.
            streaming_callback (Callable[[str, str, bool], None], optional):
                Callback function for streaming agent outputs. Parameters are
                (agent_name, chunk, is_final) where is_final indicates completion.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The formatted conversation history as output, formatted according
                 to the output_type configuration.

        Raises:
            Exception: If swarm execution fails.
        """
        try:
            # Handle interactive mode task input
            if task is None and self.interactive:
                task = self._get_interactive_task()

            current_loop = 0
            last_output = None

            # Start dashboard if in interactive mode
            if self.interactive and self.dashboard:
                self.dashboard.start(self.max_loops)
                self.dashboard.update_director_status("ACTIVE")

            while current_loop < self.max_loops:
                # Update dashboard loop counter
                if self.interactive and self.dashboard:
                    self.dashboard.update_loop(current_loop + 1)
                    self.dashboard.update_director_status(
                        "PROCESSING"
                    )

                # For the first loop, use the original task.
                # For subsequent loops, use the feedback from the previous loop as context.
                if current_loop == 0:
                    loop_task = task
                else:
                    loop_task = (
                        f"Previous loop results: {last_output}\n\n"
                        f"Original task: {task}\n\n"
                        "Based on the previous results and any feedback, continue with the next iteration of the task. "
                        "Refine, improve, or complete any remaining aspects of the analysis."
                    )

                # Execute one step of the swarm
                try:
                    last_output = self.step(
                        task=loop_task,
                        img=img,
                        streaming_callback=streaming_callback,
                        *args,
                        **kwargs,
                    )

                except Exception as e:
                    error_msg = (
                        f"[ERROR] Loop execution failed: {str(e)}"
                    )
                    logger.error(
                        f"{error_msg}\n[TRACE] Traceback: {traceback.format_exc()}\n[BUG] If this issue persists, please report it at: https://github.com/kyegomez/swarms/issues"
                    )

                current_loop += 1

                # Add loop completion marker to conversation
                self.conversation.add(
                    role="System",
                    content=f"--- Loop {current_loop}/{self.max_loops} completed ---",
                )

            # Stop dashboard if in interactive mode
            if self.interactive and self.dashboard:
                self.dashboard.update_director_status("COMPLETED")
                self.dashboard.stop()

            result = history_output_formatter(
                conversation=self.conversation, type=self.output_type
            )

            # Save conversation history after successful execution
            if self.autosave and self.swarm_workspace_dir:
                try:
                    self._save_conversation_history()
                except Exception as e:
                    logger.warning(
                        f"Failed to save conversation history: {e}"
                    )

            return result

        except Exception as e:
            # Stop dashboard on error
            if self.interactive and self.dashboard:
                self.dashboard.update_director_status("ERROR")
                self.dashboard.stop()

            # Save conversation history on error
            if self.autosave and self.swarm_workspace_dir:
                try:
                    self._save_conversation_history()
                except Exception as save_error:
                    logger.warning(
                        f"Failed to save conversation history on error: {save_error}"
                    )

            error_msg = f"[ERROR] Swarm run failed: {str(e)}"
            logger.error(
                f"{error_msg}\n[TRACE] Traceback: {traceback.format_exc()}\n[BUG] If this issue persists, please report it at: https://github.com/kyegomez/swarms/issues"
            )
            raise

    def _get_interactive_task(self) -> str:
        """
        Get task input from user in interactive mode.

        Returns:
            str: The task input from the user
        """
        if self.dashboard:
            self.dashboard.console.print(
                "\n[bold red]SWARMS CORPORATION[/bold red] - [bold white]TASK INPUT REQUIRED[/bold white]"
            )
            self.dashboard.console.print(
                "[bold cyan]Enter your task for the hierarchical swarm:[/bold cyan]"
            )

        task = input("> ")
        return task.strip()

    def feedback_director(self, outputs: list):
        """
        Generate feedback from the director based on agent outputs.

        This method creates a feedback director agent that analyzes the results
        from worker agents and provides specific, actionable feedback for improvement.
        The feedback is added to the conversation history and can be used in
        subsequent iterations.

        Args:
            outputs (list): List of outputs from worker agents that need feedback.

        Returns:
            str: The director's feedback on the agent outputs.

        Raises:
            Exception: If feedback generation fails.
        """
        try:
            task = f"History: {self.conversation.get_str()} \n\n"

            feedback_director = Agent(
                agent_name="Director",
                agent_description="Director module that provides feedback to the worker agents",
                model_name=self.director_model_name,
                max_loops=1,
                system_prompt=HIEARCHICAL_SWARM_SYSTEM_PROMPT,
            )

            output = feedback_director.run(
                task=(
                    "You are the Director. Carefully review the outputs generated by all the worker agents in the previous step. "
                    "Provide specific, actionable feedback for each agent, highlighting strengths, weaknesses, and concrete suggestions for improvement. "
                    "If any outputs are unclear, incomplete, or could be enhanced, explain exactly how. "
                    f"Your feedback should help the agents refine their work in the next iteration. "
                    f"Worker Agent Responses: {task}"
                )
            )
            self.conversation.add(
                role=self.director.agent_name, content=output
            )

            return output

        except Exception as e:
            error_msg = f"[ERROR] Feedback director failed: {str(e)}"
            logger.error(
                f"{error_msg}\n[TRACE] Traceback: {traceback.format_exc()}\n[BUG] If this issue persists, please report it at: https://github.com/kyegomez/swarms/issues"
            )

    def run_judge_agent(self, outputs: list) -> str:
        """
        Create a one-shot judge agent that scores each worker agent's output.

        Args:
            outputs (list): List of agent outputs to evaluate.

        Returns:
            str: The structured JudgeReport as a string, also added to conversation.
        """
        try:
            logger.info(
                "Running judge agent to score worker outputs..."
            )
            schema = BaseTool().base_model_to_dict(JudgeReport)

            judge = Agent(
                agent_name="JudgeAgent",
                agent_description="Evaluates and scores the quality of worker agent outputs",
                system_prompt=HIERARCHICAL_SWARM_JUDGE_PROMPT,
                model_name=self.judge_agent_model_name,
                max_loops=1,
                base_model=JudgeReport,
                tools_list_dictionary=[schema],
                output_type="final",
            )

            task = (
                f"Conversation history:\n{self.conversation.get_str()}\n\n"
                f"Agent outputs to evaluate:\n{outputs}"
            )

            result = judge.run(task=task)
            self.conversation.add(role="JudgeAgent", content=result)
            logger.info(f"Judge agent completed scoring:\n{result}")
            return result

        except Exception as e:
            logger.error(
                f"[ERROR] run_judge_agent failed: {str(e)}\n[TRACE] {traceback.format_exc()}"
            )
            return str(outputs)

    def call_single_agent(
        self,
        agent_name: str,
        task: str,
        streaming_callback: Optional[
            Callable[[str, str, bool], None]
        ] = None,
        _add_to_conversation: bool = True,
        *args,
        **kwargs,
    ):
        """
        Call a single agent by name to execute a specific task.

        This method locates an agent by name and executes the given task with
        the current conversation context. The agent's output is added to the
        conversation history for future reference.

        Args:
            agent_name (str): The name of the agent to call.
            task (str): The task to be executed by the agent.
            streaming_callback (Callable[[str, str, bool], None], optional):
                Callback function for streaming agent outputs. Parameters are
                (agent_name, chunk, is_final) where is_final indicates completion.
            *args: Additional positional arguments for the agent.
            **kwargs: Additional keyword arguments for the agent.

        Returns:
            Any: The output from the agent's execution.

        Raises:
            ValueError: If the specified agent is not found in the swarm.
            Exception: If agent execution fails.
        """
        try:
            # Find agent by name - O(1) lookup via dict
            agent = self.agent_map.get(agent_name)

            if agent is None:
                available_agents = [
                    a.agent_name
                    for a in self.agents
                    if hasattr(a, "agent_name")
                ]
                raise ValueError(
                    f"Agent '{agent_name}' not found in swarm. Available agents: {available_agents}"
                )

            # Update dashboard for agent execution
            if self.interactive and self.dashboard:
                self.dashboard.update_agent_status(
                    agent_name, "RUNNING", task, "Executing task..."
                )

            # Handle streaming callback if provided
            if streaming_callback is not None:

                def agent_streaming_callback(chunk: str):
                    """Wrapper for agent streaming callback."""
                    try:
                        if chunk is not None and chunk.strip():
                            streaming_callback(
                                agent_name, chunk, False
                            )
                    except Exception as e:
                        error_msg = f"[ERROR] Streaming callback failed for agent {agent_name}: {str(e)}"
                        logger.error(
                            f"{error_msg}\n[TRACE] Traceback: {traceback.format_exc()}"
                        )

                # Temporarily enable streaming so call_llm honours the callback
                original_streaming_on = getattr(agent, "streaming_on", False)
                agent.streaming_on = True
                try:
                    output = agent.run(
                        task=f"History: {self.conversation.get_str()} \n\n Task: {task}",
                        streaming_callback=agent_streaming_callback,
                        *args,
                        **kwargs,
                    )
                finally:
                    agent.streaming_on = original_streaming_on

                # Call completion callback
                try:
                    streaming_callback(agent_name, "", True)
                except Exception as e:
                    error_msg = f"[ERROR] Completion callback failed for agent {agent_name}: {str(e)}"
                    logger.error(
                        f"{error_msg}\n[TRACE] Traceback: {traceback.format_exc()}"
                    )
            else:
                output = agent.run(
                    task=f"History: {self.conversation.get_str()} \n\n Task: {task}",
                    *args,
                    **kwargs,
                )
            if _add_to_conversation:
                self.conversation.add(role=agent_name, content=output)

            return output

        except Exception as e:
            # Update dashboard with error status
            if self.interactive and self.dashboard:
                self.dashboard.update_agent_status(
                    agent_name, "ERROR", task, f"Error: {str(e)}"
                )

            error_msg = (
                f"[ERROR] Failed to call agent {agent_name}: {str(e)}"
            )
            logger.error(
                f"{error_msg}\n[TRACE] Traceback: {traceback.format_exc()}\n[BUG] If this issue persists, please report it at: https://github.com/kyegomez/swarms/issues"
            )

    def parse_orders(self, output):
        """
        Parse the director's output to extract plan and orders.

        This method handles various output formats from the director agent and
        extracts the plan and hierarchical orders. It supports both direct
        dictionary formats and function call formats with JSON arguments.

        Args:
            output: The raw output from the director agent.

        Returns:
            tuple: A tuple containing (plan, orders) where plan is a string
                   and orders is a list of HierarchicalOrder objects.

        Raises:
            ValueError: If the output format is unexpected or cannot be parsed.
            Exception: If parsing fails due to other errors.
        """
        try:
            import json

            # Handle different output formats from the director
            if isinstance(output, list):
                # If output is a list, look for function call data
                for item in output:
                    if isinstance(item, dict):
                        # Check if it's a conversation format with role/content
                        if "content" in item and isinstance(
                            item["content"], list
                        ):
                            for content_item in item["content"]:
                                if (
                                    isinstance(content_item, dict)
                                    and "function" in content_item
                                ):
                                    function_data = content_item[
                                        "function"
                                    ]
                                    if "arguments" in function_data:
                                        try:
                                            args = json.loads(
                                                function_data[
                                                    "arguments"
                                                ]
                                            )
                                            if (
                                                "plan" in args
                                                and "orders" in args
                                            ):
                                                plan = args["plan"]
                                                orders = [
                                                    HierarchicalOrder(
                                                        **order
                                                    )
                                                    for order in args[
                                                        "orders"
                                                    ]
                                                ]

                                                return plan, orders
                                        except json.JSONDecodeError:
                                            pass
                        # Check if it's a direct function call format
                        elif "function" in item:
                            function_data = item["function"]
                            if "arguments" in function_data:
                                try:
                                    args = json.loads(
                                        function_data["arguments"]
                                    )
                                    if (
                                        "plan" in args
                                        and "orders" in args
                                    ):
                                        plan = args["plan"]
                                        orders = [
                                            HierarchicalOrder(**order)
                                            for order in args[
                                                "orders"
                                            ]
                                        ]

                                        return plan, orders
                                except json.JSONDecodeError:
                                    pass
                # If no function call found, raise error
                raise ValueError(
                    f"Unable to parse orders from director output: {output}"
                )
            elif isinstance(output, dict):
                # Handle direct dictionary format
                if "plan" in output and "orders" in output:
                    plan = output["plan"]
                    orders = [
                        HierarchicalOrder(**order)
                        for order in output["orders"]
                    ]

                    return plan, orders
                else:
                    raise ValueError(
                        f"Missing 'plan' or 'orders' in director output: {output}"
                    )
            else:
                raise ValueError(
                    f"Unexpected output format from director: {type(output)}"
                )

        except Exception as e:
            error_msg = f"[ERROR] Failed to parse orders: {str(e)}"
            logger.error(
                f"{error_msg}\n[TRACE] Traceback: {traceback.format_exc()}\n[BUG] If this issue persists, please report it at: https://github.com/kyegomez/swarms/issues"
            )
            raise e

    def execute_orders(
        self,
        orders: list,
        streaming_callback: Optional[
            Callable[[str, str, bool], None]
        ] = None,
    ):
        """
        Execute all orders from the director's output.

        This method iterates through all hierarchical orders and calls the
        appropriate agents to execute their assigned tasks. Each agent's
        output is collected and returned as a list.

        Args:
            orders (list): List of HierarchicalOrder objects to execute.
            streaming_callback (Callable[[str, str, bool], None], optional):
                Callback function for streaming agent outputs. Parameters are
                (agent_name, chunk, is_final) where is_final indicates completion.

        Returns:
            list: List of outputs from all executed orders.

        Raises:
            Exception: If order execution fails.
        """
        try:
            if self.parallel_execution:
                max_workers = max(1, int(os.cpu_count() * 0.75))
                futures_map = {}
                results = [None] * len(orders)

                with ThreadPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    for i, order in enumerate(orders):
                        if self.interactive and self.dashboard:
                            self.dashboard.update_agent_status(
                                order.agent_name,
                                "RUNNING",
                                order.task,
                                "Processing...",
                            )
                        future = executor.submit(
                            self.call_single_agent,
                            order.agent_name,
                            order.task,
                            streaming_callback,
                            False,  # _add_to_conversation=False
                        )
                        futures_map[future] = (i, order)

                    for future in as_completed(futures_map):
                        idx, order = futures_map[future]
                        output = future.result()
                        results[idx] = output

                        if self.interactive and self.dashboard:
                            self.dashboard.update_agent_status(
                                order.agent_name,
                                "COMPLETED",
                                order.task,
                                str(output),
                            )

                # Write outputs to conversation in submission order
                for i, order in enumerate(orders):
                    if results[i] is not None:
                        self.conversation.add(
                            role=order.agent_name, content=results[i]
                        )

                return results

            else:
                outputs = []
                for i, order in enumerate(orders):
                    # Update dashboard for agent execution
                    if self.interactive and self.dashboard:
                        self.dashboard.update_agent_status(
                            order.agent_name,
                            "RUNNING",
                            order.task,
                            "Processing...",
                        )

                    output = self.call_single_agent(
                        order.agent_name,
                        order.task,
                        streaming_callback=streaming_callback,
                    )

                    # Update dashboard with completed status
                    if self.interactive and self.dashboard:
                        # Always show full output without truncation
                        output_display = str(output)

                        self.dashboard.update_agent_status(
                            order.agent_name,
                            "COMPLETED",
                            order.task,
                            output_display,
                        )

                    outputs.append(output)

                return outputs

        except Exception as e:
            error_msg = (
                "\n"
                + "=" * 60
                + "\n[SWARMS ERROR] Order Execution Failure\n"
                + "-" * 60
                + f"\nError   : {str(e)}"
                f"\nTrace   :\n{traceback.format_exc()}"
                + "-" * 60
                + "\nIf this issue persists, please report it:"
                "\n  https://github.com/kyegomez/swarms/issues"
                "\n" + "=" * 60 + "\n"
            )
            logger.error(error_msg)

    def batched_run(
        self,
        tasks: List[str],
        img: str = None,
        streaming_callback: Optional[
            Callable[[str, str, bool], None]
        ] = None,
        *args,
        **kwargs,
    ):
        """
        Execute the hierarchical swarm for multiple tasks in sequence.

        This method processes a list of tasks sequentially, running the complete
        swarm workflow for each task. Each task is processed independently with
        its own conversation context and results.

        Args:
            tasks (List[str]): List of tasks to be processed by the swarm.
            img (str, optional): Optional image input for the tasks.
            streaming_callback (Callable[[str, str, bool], None], optional):
                Callback function for streaming agent outputs. Parameters are
                (agent_name, chunk, is_final) where is_final indicates completion.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            list: List of results for each processed task.

        Raises:
            Exception: If batched execution fails.
        """
        try:
            # Initialize a list to store the results
            results = []

            # Process each task in parallel
            for task in tasks:
                result = self.run(
                    task,
                    img,
                    streaming_callback=streaming_callback,
                    *args,
                    **kwargs,
                )
                results.append(result)

            return results

        except Exception as e:
            error_msg = f"[ERROR] Batched hierarchical swarm run failed: {str(e)}"
            logger.error(
                f"{error_msg}\n[TRACE] Traceback: {traceback.format_exc()}\n[BUG] If this issue persists, please report it at: https://github.com/kyegomez/swarms/issues"
            )

    async def arun(
        self,
        task: Optional[str] = None,
        img: Optional[str] = None,
        streaming_callback: Optional[
            Callable[[str, str, bool], None]
        ] = None,
        *args,
        **kwargs,
    ) -> Any:
        """
        Async entry point that wraps the synchronous run() in asyncio.to_thread().

        Args:
            task (str, optional): The task to be processed by the swarm.
            img (str, optional): Optional image input for the agents.
            streaming_callback (Callable[[str, str, bool], None], optional):
                Callback for streaming agent outputs.
            *args: Additional positional arguments passed to run().
            **kwargs: Additional keyword arguments passed to run().

        Returns:
            Any: The same result as run().
        """
        return await asyncio.to_thread(
            self.run,
            task=task,
            img=img,
            streaming_callback=streaming_callback,
            *args,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Streaming helpers
    # ------------------------------------------------------------------

    async def _stream_agent_in_thread(
        self,
        agent: "Agent",
        task_str: str,
        img: Optional[str] = None,
    ) -> tuple:
        """Run *agent.run()* in a thread, streaming tokens via a queue.

        Returns ``(output, chunks)`` where *output* is the value returned
        by ``agent.run()`` and *chunks* is the list of streamed token
        strings.

        Yields are performed by the **caller** who drains the queue
        stored on ``self._stream_q`` between the call to this coroutine
        and its completion.
        """
        q = self._stream_q
        ev_loop = asyncio.get_running_loop()
        chunks: List[str] = []
        PHASE_DONE = self._PHASE_DONE

        def cb(chunk: str):
            if chunk is not None and chunk.strip():
                chunks.append(chunk)
                ev_loop.call_soon_threadsafe(q.put_nowait, chunk)

        original_streaming_on = getattr(agent, "streaming_on", False)
        agent.streaming_on = True
        try:
            output = await asyncio.to_thread(
                agent.run,
                task=task_str,
                img=img,
                streaming_callback=cb,
            )
        finally:
            agent.streaming_on = original_streaming_on

        # Signal this phase is done so the consumer stops draining
        ev_loop.call_soon_threadsafe(q.put_nowait, PHASE_DONE)
        return output, chunks

    async def _drain_queue_tokens(
        self,
        role: str,
        agent_name: str,
        loop_idx: int,
        with_events: bool,
    ):
        """Drain ``self._stream_q`` until ``_PHASE_DONE``, yielding events."""
        q = self._stream_q
        PHASE_DONE = self._PHASE_DONE

        while True:
            item = await q.get()
            if item is PHASE_DONE:
                break
            if with_events:
                yield {
                    "type": "token",
                    "role": role,
                    "agent": agent_name,
                    "token": item,
                    "loop": loop_idx,
                }
            else:
                yield (agent_name, item)

    async def arun_stream(
        self,
        task: Optional[str] = None,
        img: Optional[str] = None,
        with_events: bool = False,
        **kwargs,
    ):
        """Async generator that streams tokens from every phase of the
        hierarchical swarm: director planning, worker execution, and
        feedback / judge aggregation.

        Args:
            task: The task to be processed by the swarm.
            img: Optional image input for the agents.
            with_events: When False (default), yield ``(agent_name, token)``
                tuples.  When True, yield structured event dicts tagged
                with ``role`` (``director`` / ``worker`` / ``aggregator``)
                and ``loop`` index.  Event types:
                ``swarm_start``, ``director_start``, ``token``,
                ``director_end``, ``worker_start``, ``worker_end``,
                ``aggregator_start``, ``aggregator_end``, ``swarm_end``.

        Yields:
            tuple | dict: Per-token streaming items.
        """
        # Shared queue used by _stream_agent_in_thread / _drain_queue_tokens
        self._stream_q: asyncio.Queue = asyncio.Queue()
        self._PHASE_DONE = object()

        if with_events:
            yield {
                "type": "swarm_start",
                "role": "swarm",
                "loop": 0,
            }

        current_loop = 0
        last_output = None

        while current_loop < self.max_loops:
            # Build loop task
            if current_loop == 0:
                loop_task = task
            else:
                loop_task = (
                    f"Previous loop results: {last_output}\n\n"
                    f"Original task: {task}\n\n"
                    "Based on the previous results and any feedback, "
                    "continue with the next iteration of the task. "
                    "Refine, improve, or complete any remaining aspects "
                    "of the analysis."
                )

            # =============================================================
            # DIRECTOR PHASE
            # =============================================================
            director_task_str = (
                f"History: {self.conversation.get_str()} "
                f"\n\n Task: {loop_task}"
            )

            # Optional planning sub-step (non-streaming — creates a
            # throwaway agent with modified tools)
            if self.planning_enabled:
                self.director.tools_list_dictionary = None
                plan_out = await asyncio.to_thread(
                    self.setup_director_with_planning,
                    task=director_task_str,
                    img=img,
                )
                self.conversation.add(
                    role=self.director.agent_name, content=plan_out
                )
                # Refresh context after planning output was added
                director_task_str = (
                    f"History: {self.conversation.get_str()} "
                    f"\n\n Task: {loop_task}"
                )

            if with_events:
                yield {
                    "type": "director_start",
                    "role": "director",
                    "agent": self.director_name,
                    "loop": current_loop,
                }

            # Stream director in background, drain tokens here
            director_coro = self._stream_agent_in_thread(
                self.director, director_task_str, img=img,
            )
            director_task_obj = asyncio.ensure_future(director_coro)

            async for evt in self._drain_queue_tokens(
                "director", self.director_name, current_loop, with_events
            ):
                yield evt

            director_output, director_chunks = await director_task_obj
            self.conversation.add(
                role="Director", content=director_output
            )

            if with_events:
                yield {
                    "type": "director_end",
                    "role": "director",
                    "agent": self.director_name,
                    "output": str(director_output),
                    "loop": current_loop,
                }

            # Parse orders from director output
            plan, orders = self.parse_orders(director_output)

            # =============================================================
            # WORKER PHASE
            # =============================================================
            if self.parallel_execution and len(orders) > 1:
                # --- Parallel workers with interleaving ---
                worker_q: asyncio.Queue = asyncio.Queue()
                W_DONE = object()
                ev_loop = asyncio.get_running_loop()
                worker_results: Dict[str, Any] = {}
                worker_chunks: Dict[str, List[str]] = {}

                if with_events:
                    for order in orders:
                        yield {
                            "type": "worker_start",
                            "role": "worker",
                            "agent": order.agent_name,
                            "loop": current_loop,
                        }

                async def _worker_producer(order):
                    agent = self.agent_map.get(order.agent_name)
                    if agent is None:
                        return
                    w_task = (
                        f"History: {self.conversation.get_str()} "
                        f"\n\n Task: {order.task}"
                    )
                    w_chunks: List[str] = []

                    def w_cb(chunk: str):
                        if chunk is not None and chunk.strip():
                            w_chunks.append(chunk)
                            ev_loop.call_soon_threadsafe(
                                worker_q.put_nowait,
                                (order.agent_name, chunk),
                            )

                    orig = getattr(agent, "streaming_on", False)
                    agent.streaming_on = True
                    try:
                        out = await asyncio.to_thread(
                            agent.run,
                            task=w_task,
                            img=img,
                            streaming_callback=w_cb,
                        )
                    finally:
                        agent.streaming_on = orig

                    worker_results[order.agent_name] = out
                    worker_chunks[order.agent_name] = w_chunks

                producer_tasks = [
                    asyncio.create_task(_worker_producer(order))
                    for order in orders
                ]

                # Monitor completion
                async def _signal_when_done():
                    await asyncio.gather(
                        *producer_tasks, return_exceptions=True
                    )
                    ev_loop.call_soon_threadsafe(
                        worker_q.put_nowait, (W_DONE, None)
                    )

                asyncio.create_task(_signal_when_done())

                # Drain interleaved worker tokens
                while True:
                    item = await worker_q.get()
                    name, token = item
                    if name is W_DONE:
                        break
                    if with_events:
                        yield {
                            "type": "token",
                            "role": "worker",
                            "agent": name,
                            "token": token,
                            "loop": current_loop,
                        }
                    else:
                        yield (name, token)
                    await asyncio.sleep(0)

                # Write to conversation in order and emit worker_end
                for order in orders:
                    out = worker_results.get(order.agent_name)
                    if out is not None:
                        self.conversation.add(
                            role=order.agent_name, content=out
                        )
                    if with_events:
                        yield {
                            "type": "worker_end",
                            "role": "worker",
                            "agent": order.agent_name,
                            "output": "".join(
                                worker_chunks.get(order.agent_name, [])
                            ),
                            "loop": current_loop,
                        }

                worker_outputs = [
                    worker_results.get(o.agent_name) for o in orders
                ]

            else:
                # --- Sequential workers ---
                worker_outputs = []
                for order in orders:
                    agent = self.agent_map.get(order.agent_name)
                    if agent is None:
                        continue
                    w_task = (
                        f"History: {self.conversation.get_str()} "
                        f"\n\n Task: {order.task}"
                    )

                    if with_events:
                        yield {
                            "type": "worker_start",
                            "role": "worker",
                            "agent": order.agent_name,
                            "loop": current_loop,
                        }

                    w_coro = self._stream_agent_in_thread(
                        agent, w_task, img=img,
                    )
                    w_task_obj = asyncio.ensure_future(w_coro)

                    async for evt in self._drain_queue_tokens(
                        "worker",
                        order.agent_name,
                        current_loop,
                        with_events,
                    ):
                        yield evt

                    w_output, w_chunks = await w_task_obj
                    self.conversation.add(
                        role=order.agent_name, content=w_output
                    )
                    worker_outputs.append(w_output)

                    if with_events:
                        yield {
                            "type": "worker_end",
                            "role": "worker",
                            "agent": order.agent_name,
                            "output": "".join(w_chunks),
                            "loop": current_loop,
                        }

            # =============================================================
            # AGGREGATION PHASE (feedback director or judge)
            # =============================================================
            if self.agent_as_judge:
                agg_name = "JudgeAgent"
                if with_events:
                    yield {
                        "type": "aggregator_start",
                        "role": "aggregator",
                        "agent": agg_name,
                        "loop": current_loop,
                    }

                # Judge uses tool calling / structured output — run in
                # thread with streaming callback
                judge_task_str = (
                    f"Conversation history:\n"
                    f"{self.conversation.get_str()}\n\n"
                    f"Agent outputs to evaluate:\n{worker_outputs}"
                )

                schema = BaseTool().base_model_to_dict(JudgeReport)
                judge = Agent(
                    agent_name=agg_name,
                    agent_description="Evaluates and scores the quality of worker agent outputs",
                    system_prompt=HIERARCHICAL_SWARM_JUDGE_PROMPT,
                    model_name=self.judge_agent_model_name,
                    max_loops=1,
                    base_model=JudgeReport,
                    tools_list_dictionary=[schema],
                    output_type="final",
                )

                j_coro = self._stream_agent_in_thread(
                    judge, judge_task_str
                )
                j_task_obj = asyncio.ensure_future(j_coro)

                async for evt in self._drain_queue_tokens(
                    "aggregator", agg_name, current_loop, with_events,
                ):
                    yield evt

                j_output, j_chunks = await j_task_obj
                self.conversation.add(role=agg_name, content=j_output)
                last_output = j_output

                if with_events:
                    yield {
                        "type": "aggregator_end",
                        "role": "aggregator",
                        "agent": agg_name,
                        "output": "".join(j_chunks),
                        "loop": current_loop,
                    }

            elif self.director_feedback_on:
                agg_name = "Director"
                if with_events:
                    yield {
                        "type": "aggregator_start",
                        "role": "aggregator",
                        "agent": agg_name,
                        "loop": current_loop,
                    }

                fb_agent = Agent(
                    agent_name="Director",
                    agent_description="Director module that provides feedback to the worker agents",
                    model_name=self.director_model_name,
                    max_loops=1,
                    system_prompt=HIEARCHICAL_SWARM_SYSTEM_PROMPT,
                )
                fb_task_str = (
                    "You are the Director. Carefully review the outputs "
                    "generated by all the worker agents in the previous "
                    "step. Provide specific, actionable feedback for "
                    "each agent, highlighting strengths, weaknesses, "
                    "and concrete suggestions for improvement. "
                    f"Worker Agent Responses: History: "
                    f"{self.conversation.get_str()}"
                )

                fb_coro = self._stream_agent_in_thread(
                    fb_agent, fb_task_str
                )
                fb_task_obj = asyncio.ensure_future(fb_coro)

                async for evt in self._drain_queue_tokens(
                    "aggregator", agg_name, current_loop, with_events,
                ):
                    yield evt

                fb_output, fb_chunks = await fb_task_obj
                self.conversation.add(
                    role=self.director.agent_name, content=fb_output
                )
                last_output = fb_output

                if with_events:
                    yield {
                        "type": "aggregator_end",
                        "role": "aggregator",
                        "agent": agg_name,
                        "output": "".join(fb_chunks),
                        "loop": current_loop,
                    }
            else:
                last_output = worker_outputs

            current_loop += 1
            self.conversation.add(
                role="System",
                content=f"--- Loop {current_loop}/{self.max_loops} completed ---",
            )

        if with_events:
            yield {
                "type": "swarm_end",
                "role": "swarm",
                "loop": current_loop - 1,
            }

    def run_stream(
        self,
        task: Optional[str] = None,
        img: Optional[str] = None,
        with_events: bool = False,
        **kwargs,
    ):
        """Sync generator version of ``arun_stream``.

        Bridges the async generator to a sync iterator using a thread
        and a ``queue.Queue``.  Use ``arun_stream`` directly when you
        already have a running event loop (e.g. inside a FastAPI handler).

        Args:
            task: The task to be processed by the swarm.
            img: Optional image input for the agents.
            with_events: When False (default), yield ``(agent_name, token)``
                tuples.  When True, yield structured event dicts.

        Yields:
            tuple | dict: Per-token streaming items.
        """
        sync_q: _queue.Queue = _queue.Queue()
        DONE = object()
        exc_holder: List[Optional[Exception]] = [None]

        def _runner():
            async def _consume():
                async for evt in self.arun_stream(
                    task=task,
                    img=img,
                    with_events=with_events,
                    **kwargs,
                ):
                    sync_q.put(evt)

            try:
                asyncio.run(_consume())
            except Exception as e:
                exc_holder[0] = e
            finally:
                sync_q.put(DONE)

        t = threading.Thread(target=_runner, daemon=True)
        t.start()

        try:
            while True:
                item = sync_q.get()
                if item is DONE:
                    break
                yield item
        finally:
            t.join(timeout=5)

        if exc_holder[0] is not None:
            raise exc_holder[0]
