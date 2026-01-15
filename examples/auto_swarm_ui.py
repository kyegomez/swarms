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
import argparse
import os
import difflib

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

# Require explicit environment opt-in before instantiating real Agent instances
# to avoid accidental API calls / costs. Set SWARMS_ALLOW_REAL_AGENT=1 to allow.


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
    try:
        import json
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Basic validation: expect list of cases with input/expected/options
        if isinstance(data, list) and data:
            return data
    except Exception:
        pass
    return None


def compute_similarity(a: str, b: str) -> float:
    try:
        return difflib.SequenceMatcher(None, str(a), str(b)).ratio()
    except Exception:
        return 0.0

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
        scores = evaluate_agents(agents, dataset)
        show_scores_panel(scores)

        # judge and optionally improve agents
        agents = judge_and_improve(agents, scores, iteration)

    console.print(Panel(Text("Auto swarm build + evaluation completed.", style="bold red"), border_style="red"))


def evaluate_agents(agents, dataset):
    """Evaluate agents and return a dict of metrics per agent.

    Returns: {agent: {"correct": int, "total": int, "accuracy": float, "avg_similarity": float}}
    """
    scores = {}
    # if configured, try to call real Agent.run for each agent
    use_real = CONFIG.get("use_real_agents", False)
    real_agent_class = None
    if use_real:
        # require explicit opt-in via env var to avoid accidental external API usage
        if os.environ.get("SWARMS_ALLOW_REAL_AGENT") == "1":
            try:
                from swarms.structs.agent import Agent as RealAgent
                real_agent_class = RealAgent
            except Exception:
                real_agent_class = None
        else:
            console.print(Panel(Text("Real agent instantiation blocked: set SWARMS_ALLOW_REAL_AGENT=1 to allow.", style="bold yellow"), border_style="red"))
            real_agent_class = None

    for a in agents:
        correct = 0
        similarities = []
        total = len(dataset)
        if real_agent_class:
            try:
                agent = real_agent_class(agent_name=a, model_name=CONFIG.get("model"))
                for case in dataset:
                    out = agent.run(task=case["input"]) or ""
                    sim = compute_similarity(out, case.get("expected", ""))
                    similarities.append(sim)
                    if case.get("expected", "").lower() in str(out).lower():
                        correct += 1
            except Exception:
                # fallback to deterministic simulated agent
                seed = abs(hash(a)) % (2 ** 32)
                rng = random.Random(seed)
                for case in dataset:
                    resp = rng.choice(case["options"])
                    sim = compute_similarity(resp, case.get("expected", ""))
                    similarities.append(sim)
                    if resp == case["expected"]:
                        correct += 1
        else:
            seed = abs(hash(a)) % (2 ** 32)
            rng = random.Random(seed)
            for case in dataset:
                resp = rng.choice(case["options"])
                sim = compute_similarity(resp, case.get("expected", ""))
                similarities.append(sim)
                if resp == case["expected"]:
                    correct += 1

        accuracy = correct / max(1, total)
        avg_sim = sum(similarities) / max(1, len(similarities)) if similarities else 0.0
        scores[a] = {"correct": correct, "total": total, "accuracy": accuracy, "avg_similarity": avg_sim}
    return scores


def show_scores_panel(scores):
    t = Table(expand=True, show_header=True, header_style="bold red")
    t.add_column("Agent", style="red")
    t.add_column("Accuracy", style="white")
    t.add_column("AvgSim", style="white")
    # sort by accuracy then similarity
    for a, s in sorted(scores.items(), key=lambda x: (x[1]["accuracy"], x[1]["avg_similarity"]), reverse=True):
        acc_text = Text(f"{s['accuracy']:.2f}", style="bold green" if s["accuracy"] >= 0.5 else "yellow")
        sim_text = Text(f"{s['avg_similarity']:.2f}", style="bold green" if s["avg_similarity"] >= 0.5 else "yellow")
        t.add_row(Text(a, style="bold red"), acc_text, sim_text)
    console.print(Panel(t, title=Text("Evaluation Results", style="bold red"), border_style="red"))


def judge_and_improve(agents, scores, iteration=1):
    """A simple judge: keep top half agents and 'rebuild' the rest for the next loop.
    Rebuilding is simulated by renaming the agent (versioned).
    """
    if not agents:
        return agents
    # pick median as threshold
    vals = list(scores.values())
    threshold = statistics.median(vals)
    keep = [a for a in agents if scores.get(a, 0) >= threshold]
    remove = [a for a in agents if a not in keep]

    console.print(Panel(Text(f"Judge: keeping {len(keep)} agents, rebuilding {len(remove)} agents (threshold={threshold:.2f})", style="bold red"), border_style="red"))

    # rebuild removed agents as new versions
    new_agents = keep.copy()
    rebuilt_count = 0
    for _ in remove:
        rebuilt_count += 1
        new_agents.append(f"agent_rebuilt_{iteration}_{rebuilt_count}")

    # ensure list length matches original
    while len(new_agents) < len(agents):
        new_agents.append(f"agent_extra_{iteration}_{len(new_agents)}")

    return new_agents

if __name__ == "__main__":
    main()
