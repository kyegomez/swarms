"""Rich-based dashboard UI for ``HierarchicalSwarm``.

All Rich (``rich.*``) presentation logic for ``HierarchicalSwarm`` lives here
so the orchestration code in ``swarms/structs/hiearchical_swarm.py`` does not
need to know how panels, tables, or layouts are rendered.
"""

import time
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


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

        # Cached layout — rebuilt once in start(), sections updated in-place
        self._layout: Optional["Layout"] = None

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
                director_text.append(f"{i + 1}. ", style="bold cyan")
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

    def _refresh_section(self, section: str) -> None:
        """Rebuild only the named layout section and push to Live."""
        if not (
            self.live_display
            and self.is_active
            and self._layout is not None
        ):
            return
        if section == "operations_status":
            self._layout["operations_status"].update(
                self._create_status_panel()
            )
        elif section == "director_operations":
            self._layout["director_operations"].update(
                self._create_director_panel()
            )
        elif section == "agents":
            if self.detailed_view:
                self._layout["agents"].update(
                    self._create_detailed_agents_view()
                )
            else:
                self._layout["agents"].update(
                    Panel(
                        self._create_agents_table(),
                        border_style="red",
                        padding=(1, 1),
                    )
                )
        else:
            return
        self.live_display.update(self._layout)

    def start(self, max_loops: int = 1):
        """Start the dashboard display."""
        self.max_loops = max_loops
        self.start_time = time.time()
        self.is_active = True

        self._layout = self._create_dashboard_layout()
        self.live_display = Live(
            self._layout,
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

        self._refresh_section("agents")

    def update_director_status(self, status: str):
        """Update the director status."""
        self.director_status = status
        self._refresh_section("operations_status")

    def update_loop(self, current_loop: int):
        """Update the current execution loop."""
        self.current_loop = current_loop
        self._refresh_section("operations_status")

    def update_director_plan(self, plan: str):
        """Update the director's plan."""
        self.director_plan = plan
        self._refresh_section("director_operations")

    def update_director_orders(self, orders: list):
        """Update the director's orders."""
        self.director_orders = orders
        self._refresh_section("director_operations")

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
        self._refresh_section("operations_status")

    def force_refresh(self):
        """Force refresh the dashboard display."""
        for section in (
            "operations_status",
            "director_operations",
            "agents",
        ):
            self._refresh_section(section)

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
        self._refresh_section("agents")
