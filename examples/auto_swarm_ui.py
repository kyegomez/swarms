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
import random
import statistics
from swarms.tools.evaluator import load_eval_dataset, evaluate_agents, judge_and_improve

console = Console()

CONFIG = {
    "swarm_name": "Auto Swarm Builder",
    "policy": "sequential",
    "agent_count": 6,
    "model": "gpt-4o-mini",
    "timeout": "60s",
    # how many build/evaluate iterations to run
    "eval_iterations": 3,
    # attempt to run real Agent instances for evaluation if True
    "use_real_agents": False,
    # path to optional JSON eval dataset (if present it will override built-in dataset)
    "eval_dataset_path": "examples/eval_dataset.json",
}

TASK = "Collect market data, synthesize signals, and propose top 3 actions."

def make_agents(n: int, prefix: str = "agent"):
    return [f"{prefix}_{i+1}" for i in range(n)]

# tiny eval dataset (simulated test cases). In a real setup this would be
# a curated dataset with inputs & expected outputs for the swarm task.
EVAL_DATASET = [
    {"id": 1, "input": "market_up", "expected": "buy", "options": ["buy", "hold", "sell"]},
    {"id": 2, "input": "market_down", "expected": "sell", "options": ["buy", "hold", "sell"]},
    {"id": 3, "input": "flat_vol", "expected": "hold", "options": ["buy", "hold", "sell"]},
]


def load_eval_dataset(path: str):
    # deprecated: delegated to swarms.tools.evaluator.load_eval_dataset
    return load_eval_dataset(path)

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
    # keep this function generic: create visual statuses for the provided agent list
    agents = make_agents(CONFIG["agent_count"]) 
    statuses = {a: "pending" for a in agents}

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
        for a in agents:
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
    # Iterative build -> evaluate -> judge loop
    agents = make_agents(CONFIG["agent_count"]) 
    # try to load external eval dataset (JSON) if available
    external = load_eval_dataset(CONFIG.get("eval_dataset_path"))
    dataset = external if external is not None else EVAL_DATASET

    for iteration in range(1, CONFIG.get("eval_iterations", 1) + 1):
        console.print(Panel(Text(f"Iteration {iteration}: building agents...", style="bold red"), border_style="red"))
        create_agents_live()

        # run evaluation
        console.print(Panel(Text("Evaluating agents...", style="bold red"), border_style="red"))
        scores = evaluate_agents(agents, dataset, CONFIG)
        show_scores_panel(scores)

        # judge and optionally improve agents
        agents = judge_and_improve(agents, scores, CONFIG, iteration)

    console.print(Panel(Text("Auto swarm build + evaluation completed.", style="bold red"), border_style="red"))


def show_scores_panel(scores):
    t = Table(expand=True, show_header=True, header_style="bold red")
    t.add_column("Agent", style="red")
    t.add_column("Score", style="white")
    for a, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        s_text = Text(f"{s:.2f}", style="bold green" if s >= 0.5 else "yellow")
        t.add_row(Text(a, style="bold red"), s_text)
    console.print(Panel(t, title=Text("Evaluation Results", style="bold red"), border_style="red"))


# judge_and_improve delegated to swarms.tools.evaluator.judge_and_improve

if __name__ == "__main__":
    main()
