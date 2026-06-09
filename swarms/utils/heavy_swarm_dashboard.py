"""Rich-based dashboard UI for ``HeavySwarm``.

All Rich (``rich.*``) presentation logic for ``HeavySwarm`` lives here so the
orchestration code in ``swarms/structs/heavy_swarm.py`` does not need to know
how panels, tables, or progress bars are rendered. Each helper is a no-op
unless explicit dashboard methods are invoked — callers gate on
``show_dashboard`` themselves.
"""

from contextlib import contextmanager
from typing import Callable, Iterator, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

AgentConfig = Tuple[
    str, str, str, str
]  # (display_name, key, color, description)


class AgentProgressTracker:
    """Per-agent progress updater used during parallel agent execution.

    Backed by a single Rich ``Progress`` instance; each agent gets one task
    row that worker threads update through the phase methods below.
    """

    def __init__(self, progress: Progress, task_ids: dict) -> None:
        self._progress = progress
        self._task_ids = task_ids

    def _update(self, key: str, description: str) -> None:
        task_id = self._task_ids.get(key)
        if task_id is not None:
            self._progress.update(task_id, description=description)

    def initializing(self, key: str, display_name: str) -> None:
        self._update(
            key,
            f"[red]{display_name}[/red]: INITIALIZING ••••••••",
        )

    def processing(self, key: str, display_name: str) -> None:
        self._update(
            key,
            f"[red]{display_name}[/red]: PROCESSING QUERY ••••••••••••••",
        )

    def executing(self, key: str, display_name: str) -> None:
        self._update(
            key,
            f"[red]{display_name}[/red]: EXECUTING ••••••••••••••••••••",
        )

    def responding(self, key: str, display_name: str) -> None:
        self._update(
            key,
            f"[white]{display_name}[/white]: GENERATING RESPONSE "
            "••••••••••••••••••••••••••",
        )

    def complete(self, key: str, display_name: str) -> None:
        self._update(
            key,
            f"[bold white]{display_name}[/bold white]: ✅ COMPLETE! "
            "••••••••••••••••••••••••••••••••",
        )

    def error(self, key: str, display_name: str) -> None:
        self._update(
            key,
            f"[bold red]{display_name}[/bold red]: ❌ ERROR! "
            "••••••••••••••••••••••••••••••••",
        )

    def timeout(self, key: str, display_name: str) -> None:
        self._update(
            key,
            f"[bold red]{display_name}[/bold red]: ⏰ TIMEOUT! "
            "••••••••••••••••••••••••••••••••",
        )


class HeavySwarmDashboard:
    """Encapsulates every Rich-rendered surface used by ``HeavySwarm``."""

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()

    # ----- Static panels ---------------------------------------------------

    def show_config(
        self,
        *,
        name: str,
        description: str,
        timeout: int,
        question_model: str,
        worker_model: str,
        max_workers: int,
    ) -> None:
        info_table = Table(
            title="⚡ HEAVYSWARM CONFIGURATION",
            show_header=True,
            header_style="bold red",
        )
        info_table.add_column("Parameter", style="white", width=25)
        info_table.add_column("Value", style="bright_white", width=40)

        info_table.add_row("Swarm Name", name)
        info_table.add_row("Description", description)
        info_table.add_row("Timeout", f"{timeout}s")
        info_table.add_row("Question Model", question_model)
        info_table.add_row("Worker Model", worker_model)
        info_table.add_row("Max Workers", str(max_workers))

        self.console.print(
            Panel(
                info_table,
                title="[bold red]HEAVYSWARM SYSTEM[/bold red]",
                border_style="red",
            )
        )
        self.console.print()

    def show_reliability_complete(self) -> None:
        self.console.print(
            Panel(
                "[bold red]✅ HEAVYSWARM RELIABILITY CHECK COMPLETE[/bold red]\n"
                "[white]All systems validated and ready for operation[/white]",
                title="[bold red]SYSTEM STATUS[/bold red]",
                border_style="red",
            )
        )
        self.console.print()

    def show_task_init(self, task: str) -> None:
        self.console.print(
            Panel(
                f"[bold red]⚡ Completing Task[/bold red]\n"
                f"[white]Task: {task}[/white]",
                title="[bold red]Initializing HeavySwarm[/bold red]",
                border_style="red",
            )
        )
        self.console.print()

    def show_agent_launch_phase(
        self, agent_count: int, agent_label: str
    ) -> None:
        self.console.print(
            Panel(
                f"[bold red]⚡ LAUNCHING SPECIALIZED AGENTS[/bold red]\n"
                f"[white]Executing {agent_count} {agent_label} in parallel "
                f"for comprehensive analysis[/white]",
                title="[bold red]AGENT EXECUTION PHASE[/bold red]",
                border_style="red",
            )
        )

    def show_synthesis_complete(self) -> None:
        self.console.print(
            Panel(
                "[bold red]⚡ HEAVYSWARM ANALYSIS COMPLETE![/bold red]\n"
                "[white]Comprehensive multi-agent analysis delivered "
                "successfully[/white]",
                title="[bold red]MISSION ACCOMPLISHED[/bold red]",
                border_style="red",
            )
        )
        self.console.print()

    def show_execution_complete(
        self, agent_count: int, synth_label: str
    ) -> None:
        self.console.print(
            Panel(
                "[bold red]⚡ ALL AGENTS COMPLETED SUCCESSFULLY![/bold red]\n"
                f"[white]Results from all {agent_count} specialized agents "
                f"are ready for {synth_label}[/white]",
                title="[bold red]EXECUTION COMPLETE[/bold red]",
                border_style="red",
            )
        )
        self.console.print()

    # ----- Progress contexts ----------------------------------------------

    @contextmanager
    def reliability_progress(
        self, total: int = 4
    ) -> Iterator[Callable[[str], None]]:
        """Yield a ``step(description)`` callable that advances the bar by 1."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=self.console,
        ) as progress:
            task = progress.add_task(
                "[red]RUNNING RELIABILITY CHECKS...", total=total
            )

            def step(description: str) -> None:
                progress.update(
                    task, advance=1, description=description
                )

            yield step

    @contextmanager
    def question_generation_progress(
        self,
    ) -> Iterator[Tuple[Callable[[], None], Callable[[], None]]]:
        """Yield (start, finish) callbacks bracketing the question phase."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=self.console,
        ) as progress:
            task_gen = progress.add_task(
                "[red]⚡ GENERATING SPECIALIZED QUESTIONS...",
                total=100,
            )

            def start() -> None:
                progress.update(task_gen, advance=30)

            def finish() -> None:
                progress.update(
                    task_gen,
                    advance=70,
                    description="[white]✓ QUESTIONS GENERATED SUCCESSFULLY!",
                )

            yield start, finish

    @contextmanager
    def synthesis_progress(
        self, synth_name: str
    ) -> Iterator[Callable[[str], None]]:
        """Yield an ``update(stage)`` callback for the synthesis phase.

        ``stage`` is one of: ``"integrating"``, ``"summarizing"``,
        ``"generating"``, ``"complete"``.
        """
        dots = "••••••••••••••••••••••••••••••••"
        stages = {
            "integrating": (
                f"[red]{synth_name}: INTEGRATING AGENT RESULTS {dots}"
            ),
            "summarizing": (
                f"[red]{synth_name}: Summarizing Results {dots}"
            ),
            "generating": (
                f"[white]{synth_name}: GENERATING FINAL REPORT {dots}"
            ),
            "complete": (
                f"[bold white]{synth_name}: COMPLETE! {dots}"
            ),
        }

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            synthesis_task = progress.add_task(
                f"[red]{synth_name}: SYNTHESIZING COMPREHENSIVE ANALYSIS {dots}",
                total=None,
            )

            def update(stage: str) -> None:
                description = stages.get(stage)
                if description is not None:
                    progress.update(
                        synthesis_task, description=description
                    )

            yield update

    @contextmanager
    def agent_progress_tracker(
        self, agent_configs: List[AgentConfig]
    ) -> Iterator[AgentProgressTracker]:
        """Yield an ``AgentProgressTracker`` for live per-agent phase updates."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task_ids = {}
            for (
                display_name,
                agent_key,
                color,
                _description,
            ) in agent_configs:
                task_ids[agent_key] = progress.add_task(
                    f"[{color}]{display_name}[/{color}]: INITIALIZING",
                    total=None,
                )
            yield AgentProgressTracker(progress, task_ids)
