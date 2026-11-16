"""
Benchmark GraphWorkflow against LangGraph on identical graphs.

Every node sleeps for a fixed delay instead of calling a model, so the
numbers measure scheduling and framework overhead only. LangGraph runs
are skipped when langgraph is not installed.

Run from the repo root:
    PYTHONPATH=. python tests/benchmark_graph_workflow.py
    PYTHONPATH=. python tests/benchmark_graph_workflow.py --repeat 5
"""

import argparse
import asyncio
import os
import statistics
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Annotated

os.environ.setdefault("SWARMS_TELEMETRY_ON", "False")

from loguru import logger

from swarms.structs.graph_workflow import GraphWorkflow

# Per-run INFO lines would bury the table.
logger.disable("swarms")

try:
    from langgraph.graph import END, START, StateGraph
    from typing_extensions import TypedDict

    LANGGRAPH = True
except ImportError:
    LANGGRAPH = False

Graph = tuple[dict[str, float], list[tuple[str, str]]]

_peak_threads = 0
_peak_lock = threading.Lock()


def _note_threads() -> None:
    global _peak_threads
    with _peak_lock:
        _peak_threads = max(_peak_threads, threading.active_count())


def work(name: str, delay: float) -> str:
    """
    Stand in for one model call.

    Args:
        name (str): Node name.
        delay (float): Seconds to sleep.

    Returns:
        str: A short output naming the node.
    """
    _note_threads()
    time.sleep(delay)
    return f"{name} done"


async def awork(name: str, delay: float) -> str:
    """
    Async stand in for one model call.

    Args:
        name (str): Node name.
        delay (float): Seconds to sleep.

    Returns:
        str: A short output naming the node.
    """
    _note_threads()
    await asyncio.sleep(delay)
    return f"{name} done"


class StubAgent:
    """Agent stand in whose run sleeps instead of calling a model."""

    def __init__(self, name: str, delay: float):
        self.agent_name = name
        self.delay = delay

    def run(
        self, task=None, img=None, messages=None, *args, **kwargs
    ):
        return work(self.agent_name, self.delay)


def staggered_chains(chains: int, slow: float, fast: float) -> Graph:
    """
    Parallel chains where chain i is slow at step i.

    Args:
        chains (int): Number of chains, also the length of each.
        slow (float): Delay of the one slow node per chain.
        fast (float): Delay of every other node.

    Returns:
        Graph: Node delays and edges.
    """
    delays, edges = {}, []
    for c in range(chains):
        for s in range(chains):
            delays[f"c{c}s{s}"] = slow if s == c else fast
            if s:
                edges.append((f"c{c}s{s - 1}", f"c{c}s{s}"))
    return delays, edges


def fan_out(width: int, delay: float) -> Graph:
    """
    One entry node, width workers, one aggregator.

    Args:
        width (int): Number of parallel workers.
        delay (float): Delay of each worker.

    Returns:
        Graph: Node delays and edges.
    """
    delays = {"entry": 0.0, "agg": 0.0}
    edges = []
    for i in range(width):
        delays[f"w{i}"] = delay
        edges += [("entry", f"w{i}"), (f"w{i}", "agg")]
    return delays, edges


def chain(length: int) -> Graph:
    """
    A single chain of zero-delay nodes, to expose per-node overhead.

    Args:
        length (int): Number of nodes.

    Returns:
        Graph: Node delays and edges.
    """
    delays = {f"n{i}": 0.0 for i in range(length)}
    edges = [(f"n{i - 1}", f"n{i}") for i in range(1, length)]
    return delays, edges


def critical_path(graph: Graph) -> float:
    """
    Longest delay-weighted path, the best any scheduler can do.

    Args:
        graph (Graph): Node delays and edges.

    Returns:
        float: Seconds along the slowest dependency chain.
    """
    delays, edges = graph
    preds = defaultdict(list)
    for s, t in edges:
        preds[t].append(s)
    finish: dict[str, float] = {}

    def done(n: str) -> float:
        if n not in finish:
            finish[n] = delays[n] + max(
                (done(p) for p in preds[n]), default=0.0
            )
        return finish[n]

    return max(done(n) for n in delays)


def build_swarms(graph: Graph) -> GraphWorkflow:
    """
    Build and compile a GraphWorkflow of stub agents.

    Args:
        graph (Graph): Node delays and edges.

    Returns:
        GraphWorkflow: The compiled workflow.
    """
    delays, edges = graph
    wf = GraphWorkflow(name="bench", verbose=False)
    wf.add_nodes([StubAgent(n, d) for n, d in delays.items()])
    for s, t in edges:
        wf.add_edge(s, t)
    wf.compile()
    return wf


def build_langgraph(graph: Graph, use_async: bool):
    """
    Build and compile the same graph in LangGraph.

    Args:
        graph (Graph): Node delays and edges.
        use_async (bool): Give every node an async body.

    Returns:
        The compiled LangGraph app.
    """
    delays, edges = graph

    def merge(a: dict, b: dict) -> dict:
        return {**a, **b}

    class State(TypedDict):
        out: Annotated[dict, merge]

    def make(name: str, delay: float):
        if use_async:

            async def node(state):
                return {"out": {name: await awork(name, delay)}}

        else:

            def node(state):
                return {"out": {name: work(name, delay)}}

        return node

    g = StateGraph(State)
    for n, d in delays.items():
        g.add_node(n, make(n, d))
    preds, succs = defaultdict(list), defaultdict(list)
    for s, t in edges:
        preds[t].append(s)
        succs[s].append(t)
    for n in delays:
        if not preds[n]:
            g.add_edge(START, n)
        elif len(preds[n]) == 1:
            g.add_edge(preds[n][0], n)
        else:
            # The list form waits for every predecessor, as GraphWorkflow does.
            g.add_edge(preds[n], n)
        if not succs[n]:
            g.add_edge(n, END)
    return g.compile()


def measure(
    call: Callable[[], object], repeat: int
) -> tuple[float, int]:
    """
    Time a call several times.

    Args:
        call (Callable[[], object]): The run to time.
        repeat (int): Number of timed runs.

    Returns:
        tuple[float, int]: Median seconds and peak thread count.
    """
    global _peak_threads
    _peak_threads = 0
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        call()
        times.append(time.perf_counter() - start)
    return statistics.median(times), _peak_threads


def run_scenario(
    label: str, graph: Graph, repeat: int, use_async: bool = False
) -> list[tuple[str, str, float, float, int]]:
    """
    Run one graph in both frameworks.

    Args:
        label (str): Scenario name for the table.
        graph (Graph): Node delays and edges.
        repeat (int): Number of timed runs per framework.
        use_async (bool): Use arun and ainvoke instead of run and invoke.

    Returns:
        list[tuple[str, str, float, float, int]]: One row per framework.
    """
    ideal = critical_path(graph)
    rows = []

    wf = build_swarms(graph)
    if use_async:
        call = lambda: asyncio.run(wf.arun("task"))
    else:
        call = lambda: wf.run("task")
    wall, threads = measure(call, repeat)
    rows.append((label, "swarms", wall, ideal, threads))

    if LANGGRAPH:
        app = build_langgraph(graph, use_async)
        config = {
            "max_concurrency": len(graph[0]),
            "recursion_limit": len(graph[0]) + 10,
        }
        state = {"out": {}}
        if use_async:
            call = lambda: asyncio.run(app.ainvoke(state, config))
        else:
            call = lambda: app.invoke(state, config)
        wall, threads = measure(call, repeat)
        rows.append((label, "langgraph", wall, ideal, threads))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    repeat = parser.parse_args().repeat

    scenarios = [
        ("staggered 4x4", staggered_chains(4, 0.4, 0.05), False),
        ("fan-out 10", fan_out(10, 0.2), False),
        ("fan-out 100", fan_out(100, 0.2), False),
        ("fan-out 1000", fan_out(1000, 0.2), False),
        ("fan-out 1000 async", fan_out(1000, 0.2), True),
        ("chain 200 x 0s", chain(200), False),
    ]

    print(
        f"{'scenario':<20}{'framework':<11}{'wall s':>9}"
        f"{'ideal s':>9}{'overhead s':>12}{'threads':>9}"
    )
    for label, graph, use_async in scenarios:
        for name, fw, wall, ideal, threads in run_scenario(
            label, graph, repeat, use_async
        ):
            print(
                f"{name:<20}{fw:<11}{wall:>9.3f}{ideal:>9.3f}"
                f"{wall - ideal:>12.3f}{threads:>9}"
            )
    if not LANGGRAPH:
        print("\nlanggraph is not installed, so only swarms ran.")


if __name__ == "__main__":
    main()
