import argparse
import importlib
import operator
import os
import statistics
import subprocess
import sys
import threading
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Annotated, Callable


class StubModel:
    def __init__(self, delay: float, threads: list):
        self.delay = delay
        self.threads = threads

    def run(self, task=None, *args, **kwargs) -> str:
        self.threads.append(threading.active_count())
        time.sleep(self.delay)
        return "done"


def timed(call) -> float:
    start = time.perf_counter()
    call()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Framework overhead with a stubbed model."
    )
    parser.add_argument("--agents", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    os.environ.setdefault("SWARMS_TELEMETRY_ON", "False")

    columns = []
    for name in FRAMEWORKS:
        with ProcessPoolExecutor(1) as pool:
            try:
                column = pool.submit(measure, name, args).result()
            except ModuleNotFoundError:
                column = [(r, "not installed") for r, _ in columns[0]]
            columns.append(column)

    print(f"| benchmark | {' | '.join(FRAMEWORKS)} |")
    print("|---" * (len(FRAMEWORKS) + 1) + "|")
    for cells in zip(*columns):
        values = " | ".join(str(value) for _, value in cells)
        print(f"| {cells[0][0]} | {values} |")


def measure(name: str, args: argparse.Namespace) -> list:
    module, fanout = FRAMEWORKS[name]
    threads: list = []
    model = StubModel(args.delay, threads)
    build = partial(fanout, model, args.agents)
    importlib.import_module(module)

    def median(call) -> float:
        return statistics.median(
            timed(call) for _ in range(args.repeat)
        )

    command = [sys.executable, "-c", f"import {module}"]
    import_s = median(lambda: subprocess.run(command, check=True))
    tracemalloc.start()
    run = build()
    agent_kb = tracemalloc.get_traced_memory()[0] / 1024 / args.agents
    run()
    peak_mb = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    build_ms = timed(build) * 1000 / args.agents
    wall = median(run)

    rows = [
        ("cold import (s)", import_s),
        ("construction per agent, workflow included (ms)", build_ms),
        ("memory per agent, workflow included (KB)", agent_kb),
        (f"{args.agents}-agent fan-out wall (s)", wall),
        ("fan-out wall minus model delay (s)", wall - args.delay),
        ("fan-out peak memory, agents included (MB)", peak_mb),
        ("fan-out peak threads", max(threads)),
    ]
    return [(row, round(value, 3)) for row, value in rows]


def swarms_fanout(model: StubModel, agents: int) -> Callable:
    from swarms import Agent, ConcurrentWorkflow

    workflow = ConcurrentWorkflow(
        agents=[
            Agent(agent_name=f"agent-{i}", llm=model, print_on=False)
            for i in range(agents)
        ],
        on_error="raise",
    )
    return lambda: workflow.run("task")


def langgraph_fanout(model: StubModel, agents: int) -> Callable:
    from langgraph.graph import START, StateGraph

    graph = StateGraph(Annotated[list, operator.add])
    for i in range(agents):
        graph.add_node(f"agent-{i}", lambda _: [model.run()])
        graph.add_edge(START, f"agent-{i}")
    app = graph.compile()
    return lambda: app.invoke([], {"max_concurrency": agents})


FRAMEWORKS = {
    "swarms": ("swarms", swarms_fanout),
    "langgraph": ("langgraph.graph", langgraph_fanout),
}


if __name__ == "__main__":
    main()
