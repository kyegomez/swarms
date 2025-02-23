import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import psutil
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from swarms.structs.agent import Agent

try:
    import pynvml

    pynvml.nvmlInit()
    GPU_ENABLED = True
except ImportError:
    GPU_ENABLED = False


@dataclass
class SwarmMetadata:
    name: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = None
    type: Optional[str] = None  # hierarchical, parallel, sequential
    created_at: Optional[datetime] = None
    author: Optional[str] = None
    tags: Optional[List[str]] = None
    primary_objective: Optional[str] = None
    secondary_objectives: Optional[List[str]] = None

    def __post_init__(self):
        self.tags = self.tags or []
        self.secondary_objectives = self.secondary_objectives or []
        self.created_at = self.created_at or datetime.now()


class SwarmVisualizationRich:
    def __init__(
        self,
        swarm_metadata: SwarmMetadata,
        agents: List[Agent],
        update_resources: bool = True,
        refresh_rate: float = 0.1,
    ):
        """
        Initializes the visualizer with a list of agents.

        Args:
            swarm_metadata (SwarmMetadata): Metadata for the swarm.
            agents (List[Agent]): List of root agents.
            update_resources (bool): Whether to update system resource stats.
            refresh_rate (float): Refresh rate for the live visualization.
        """
        self.swarm_metadata = swarm_metadata
        self.agents = agents
        self.update_resources = update_resources
        self.refresh_rate = refresh_rate
        self.console = Console()
        self.live = None
        # A dictionary mapping agent names to list of output messages
        self.output_history: Dict[str, List[Dict[str, Any]]] = {}

        # System monitoring
        self.cores_available = 0
        self.memory_usage = "N/A"
        self.gpu_power = "N/A"
        self.start_time = datetime.now()

        if self.update_resources:
            self._update_resource_stats()

    def _format_uptime(self) -> str:
        """Formats the swarm's uptime."""
        delta = datetime.now() - self.start_time
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _build_agent_tree(self, agents: List[Agent]) -> Tree:
        """
        Builds a detailed tree visualization for a list of agents.

        Args:
            agents (List[Agent]): The list of root agents.

        Returns:
            Tree: A rich Tree object displaying agent metadata.
        """
        tree = Tree("[bold underline]Agents[/bold underline]")
        for agent in agents:
            self._add_agent_to_tree(agent, tree)
        return tree

    def _add_agent_to_tree(self, agent: Agent, tree: Tree) -> None:
        """
        Recursively adds an agent and its children to the given tree.

        Args:
            agent (Agent): The agent to add.
            tree (Tree): The tree to update.
        """
        agent_info = [
            f"[bold cyan]{agent.name}[/bold cyan]",
            f"[yellow]Role:[/yellow] {agent.role}",
        ]

        # # Add any custom metadata from the agent (if available)
        # for key, value in getattr(agent, "metadata", {}).items():
        #     agent_info.append(f"[white]{key}:[/white] {value}")

        # # Parameters summary if available
        # parameters = getattr(agent, "parameters", {})
        # if parameters:
        #     param_summary = ", ".join(f"{k}: {v}" for k, v in parameters.items())
        #     agent_info.append(f"[white]Parameters:[/white] {param_summary}")

        node_text = "\n".join(agent_info)
        branch = tree.add(node_text)
        for child in getattr(agent, "children", []):
            self._add_agent_to_tree(child, branch)

    def _count_agents(self, agents: List[Agent]) -> int:
        """
        Recursively counts total number of agents from a list of root agents.

        Args:
            agents (List[Agent]): List of agents.

        Returns:
            int: Total count of agents including children.
        """
        return len(agents)

    def _create_unified_info_panel(self) -> Panel:
        """
        Creates a unified panel showing swarm metadata and agents' metadata.
        """
        info_layout = Layout()
        info_layout.split_column(
            Layout(name="metadata", size=15),
            Layout(name="architecture"),
        )

        total_agents = self._count_agents(self.agents)

        # Metadata section
        metadata_table = Table.grid(padding=1, expand=True)
        metadata_table.add_column("Label", style="bold cyan")
        metadata_table.add_column("Value", style="white")

        # Update system resources if needed
        if self.update_resources:
            self._update_resource_stats()

        # Wrap the description text properly
        description_text = Text(
            self.swarm_metadata.description or "", style="italic"
        )
        description_text.wrap(self.console, width=60, overflow="fold")

        metadata_table.add_row(
            "Swarm Name", self.swarm_metadata.name or "N/A"
        )
        metadata_table.add_row("Description", description_text)
        metadata_table.add_row(
            "Version", self.swarm_metadata.version or "N/A"
        )
        metadata_table.add_row("Total Agents", str(total_agents))
        metadata_table.add_row(
            "Author", self.swarm_metadata.author or "N/A"
        )
        metadata_table.add_row(
            "System",
            f"CPU: {self.cores_available} cores | Memory: {self.memory_usage}",
        )
        metadata_table.add_row(
            "Primary Objective",
            self.swarm_metadata.primary_objective or "N/A",
        )

        info_layout["metadata"].update(metadata_table)

        # Architecture section with the agent tree
        architecture_tree = self._build_agent_tree(self.agents)
        info_layout["architecture"].update(architecture_tree)

        return Panel(
            info_layout,
            title="[bold]Swarm Information & Architecture[/bold]",
        )

    def _create_outputs_panel(self) -> Panel:
        """
        Creates a panel that displays stacked message history for all agents.
        """
        all_messages = []

        def collect_agent_messages(agent: Agent):
            """Recursively collect messages from an agent and its children."""
            messages = self.output_history.get(agent.name, [])
            for msg in messages:
                all_messages.append(
                    {
                        "agent": agent.name,
                        "time": msg["time"],
                        "content": msg["content"],
                        "style": msg["style"],
                    }
                )
            for child in getattr(agent, "children", []):
                collect_agent_messages(child)

        # Collect messages from every root agent
        for agent in self.agents:
            collect_agent_messages(agent)

        # Sort messages by timestamp
        all_messages.sort(key=lambda x: x["time"])

        messages_container = []
        for msg in all_messages:
            message_text = Text()
            message_text.append(f"[{msg['time']}] ", style="dim")
            message_text.append(
                f"{msg['agent']}: ", style="bold cyan"
            )
            message_text.append(msg["content"], style=msg["style"])
            messages_container.append(message_text)

        if messages_container:
            final_text = Text("\n").join(messages_container)
        else:
            final_text = Text("No messages yet...", style="dim")

        return Panel(
            final_text,
            title="[bold]Agent Communication Log[/bold]",
            border_style="green",
            padding=(1, 2),
        )

    def _update_resource_stats(self):
        """Updates system resource statistics."""
        self.cores_available = psutil.cpu_count(logical=True)
        mem_info = psutil.virtual_memory()
        total_gb = mem_info.total / (1024**3)
        used_gb = mem_info.used / (1024**3)
        self.memory_usage = f"{used_gb:.1f}GB / {total_gb:.1f}GB ({mem_info.percent}%)"

        if GPU_ENABLED:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                gpu_info = []
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode()
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    usage = (mem.used / mem.total) * 100
                    gpu_info.append(f"{name}: {usage:.1f}%")
                self.gpu_power = " | ".join(gpu_info)
            except Exception as e:
                self.gpu_power = f"GPU Error: {str(e)}"
        else:
            self.gpu_power = "No GPU detected"

    async def stream_output(
        self,
        agent: Agent,
        text: str,
        title: Optional[str] = None,
        style: str = "bold cyan",
        delay: float = 0.05,
        by_word: bool = False,
    ):
        """
        Streams output for a specific agent with token-by-token animation.

        Args:
            agent (Agent): The agent whose output is being streamed.
            text (str): The text to stream.
            title (Optional[str]): Custom title for the output panel.
            style (str): Style for the output text.
            delay (float): Delay between tokens.
            by_word (bool): If True, stream word by word instead of character by character.
        """
        display_text = Text(style=style)
        current_output = ""

        tokens = text.split() if by_word else text
        title = title or f"{agent.name} Output"

        for token in tokens:
            token_with_space = token + (" " if by_word else "")
            current_output += token_with_space
            display_text.append(token_with_space)

            if agent.name not in self.output_history:
                self.output_history[agent.name] = []

            if token == tokens[-1]:
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.output_history[agent.name].append(
                    {
                        "time": timestamp,
                        "content": current_output,
                        "style": style,
                    }
                )

            if self.live:
                self.live.update(self._create_layout())
            await asyncio.sleep(delay)

    def log_agent_output(self, agent: Agent, text: str):
        asyncio.create_task(
            self.stream_output(
                agent=agent,
                text=text,
                title=f"{agent.name} Output {agent.max_loops}",
            )
        )

    async def print_progress(
        self,
        description: str,
        task_fn: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Displays a progress spinner while executing a task.

        Args:
            description (str): Task description.
            task_fn (Callable): Function to execute.
            *args (Any): Arguments for task_fn.
            **kwargs (Any): Keyword arguments for task_fn.

        Returns:
            Any: The result of task_fn.
        """
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
        )

        try:
            with progress:
                task = progress.add_task(description, total=None)
                result = await task_fn(*args, **kwargs)
                progress.update(task, completed=True)
                return result
        except Exception as e:
            progress.stop()
            raise e

    def _create_layout(self) -> Layout:
        """Creates the main visualization layout."""
        layout = Layout()
        layout.split_row(
            Layout(name="info", ratio=2),
            Layout(name="outputs", ratio=3),
        )

        layout["info"].update(self._create_unified_info_panel())
        layout["outputs"].update(self._create_outputs_panel())

        return layout

    async def start(self):
        """Starts the visualization with live updates."""
        with Live(
            self._create_layout(),
            refresh_per_second=int(1 / self.refresh_rate),
        ) as self.live:
            while True:

                def process_agent_streams(agent: Agent):
                    while not agent.output_stream.empty():
                        new_output = agent.output_stream.get()
                        asyncio.create_task(
                            self.stream_output(agent, new_output)
                        )
                    for child in getattr(agent, "children", []):
                        process_agent_streams(child)

                # Process streams for each root agent
                for agent in self.agents:
                    process_agent_streams(agent)
                await asyncio.sleep(self.refresh_rate)


# # Example usage
# if __name__ == "__main__":
#     # Create swarm metadata
#     swarm_metadata = SwarmMetadata(
#         name="Financial Advisory Swarm",
#         description="Intelligent swarm for financial analysis and advisory",
#         version="1.0.0",
#         type="hierarchical",
#         created_at=datetime.now(),
#         author="AI Research Team",
#         # tags=["finance", "analysis", "advisory"],
#         primary_objective="Provide comprehensive financial analysis and recommendations",
#         secondary_objectives=[
#             "Monitor market trends",
#             "Analyze competitor behavior",
#             "Generate investment strategies",
#         ],
#     )

#     # Create agent hierarchy with detailed parameters
#     analyst = Agent(
#         name="Financial Analyst",
#         role="Analysis",
#         description="Analyzes financial data and market trends",
#         agent_type="LLM",
#         capabilities=[
#             "data analysis",
#             "trend detection",
#             "risk assessment",
#         ],
#         parameters={"model": "gpt-4", "temperature": 0.7},
#         metadata={
#             "specialty": "Market Analysis",
#             "confidence_threshold": "0.85",
#         },
#     )

#     researcher = Agent(
#         name="Market Researcher",
#         role="Research",
#         description="Conducts market research and competitor analysis",
#         agent_type="Neural",
#         capabilities=[
#             "competitor analysis",
#             "market sentiment",
#             "trend forecasting",
#         ],
#         parameters={"batch_size": 32, "learning_rate": 0.001},
#         metadata={
#             "data_sources": "Bloomberg, Reuters",
#             "update_frequency": "1h",
#         },
#     )

#     advisor = Agent(
#         name="Investment Advisor",
#         role="Advisory",
#         description="Provides investment recommendations",
#         agent_type="Hybrid",
#         capabilities=[
#             "portfolio optimization",
#             "risk management",
#             "strategy generation",
#         ],
#         parameters={
#             "risk_tolerance": "moderate",
#             "time_horizon": "long",
#         },
#         metadata={
#             "certification": "CFA Level 3",
#             "specialization": "Equity",
#         },
#         children=[analyst, researcher],
#     )

#     # Create visualization
#     viz = SwarmVisualizationRich(
#         swarm_metadata=swarm_metadata,
#         root_agent=advisor,
#         refresh_rate=0.1,
#     )

#     # Example of streaming output simulation
#     async def simulate_outputs():
#         await viz.stream_output(
#             advisor,
#             "Analyzing market conditions...\nGenerating investment advice...",
#         )
#         await viz.stream_output(
#             analyst,
#             "Processing financial data...\nIdentifying trends...",
#         )
#         await viz.stream_output(
#             researcher,
#             "Researching competitor movements...\nAnalyzing market share...",
#         )

#     # Run the visualization
#     async def main():
#         viz_task = asyncio.create_task(viz.start())
#         await simulate_outputs()
#         await viz_task

#     asyncio.run(main())
