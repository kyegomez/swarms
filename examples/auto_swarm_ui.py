#!/usr/bin/env python3
"""Auto Swarm Builder UI (Rich)

Displays an Arasaka-styled (black & red) terminal UI that:
- shows configuration
- shows the task
- animates agents being created one-by-one

Usage: python3 examples/auto_swarm_ui.py
Install dependency: pip install rich
"""
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.align import Align
import time

console = Console()

CONFIG = {
    "swarm_name": "Auto Swarm Builder",
    "policy": "sequential",
    "agent_count": 6,
    "model": "gpt-4o-mini",
    "timeout": "60s",
}

TASK = "Collect market data, synthesize signals, and propose top 3 actions."

AGENTS = [f"agent_{i+1}" for i in range(CONFIG["agent_count"]) ]

def title_text(text: str) -> Text:
    return Text(text, style="bold red on black")

def show_configuration() -> None:
    table = Table.grid(padding=(0,1))
    table.add_column(justify="right", style="red", ratio=30)
    table.add_column(justify="left", style="white", ratio=70)
    for k, v in CONFIG.items():
        table.add_row(Text(k, style="bold red"), Text(str(v), style="white"))
    panel = Panel(table, title=Text("Configuration", style="bold red"), border_style="red")
    console.print(panel)

def show_task() -> None:
    panel = Panel(Text(TASK, style="bold red"), title=Text("Task", style="bold red"), border_style="red")
    console.print(panel)

def create_agents_live() -> None:
    statuses = {a: "pending" for a in AGENTS}

    def render() -> Panel:
        t = Table(expand=True, show_header=True, header_style="bold red")
        t.add_column("Agent", style="red", no_wrap=True)
        t.add_column("Status", style="white")
        for a, s in statuses.items():
            if s == "ready":
                s_text = Text("✔ ready", style="bold green")
            elif s == "creating":
                s_text = Text("… creating", style="yellow")
            else:
                s_text = Text(s, style="white")
            t.add_row(Text(a, style="bold red"), s_text)
        return Panel(t, title=Text("Agents", style="bold red"), border_style="red")

    with Live(render(), refresh_per_second=10, console=console) as live:
        for a in AGENTS:
            statuses[a] = "creating"
            live.update(render())
            time.sleep(0.8)
            # simulate some setup steps
            statuses[a] = "finalizing"
            live.update(render())
            time.sleep(0.5)
            statuses[a] = "ready"
            live.update(render())
            time.sleep(0.25)

    console.print(Panel(Text("All agents created.", style="bold red"), border_style="red"))

def main() -> None:
    console.clear()
    console.print(Align.center(title_text("ARASAKA — AUTO SWARM BUILDER")))
    console.print()
    show_configuration()
    show_task()
    create_agents_live()

if __name__ == "__main__":
    main()
