import argparse
import os
import statistics
import subprocess
import sys
import threading
import time
import tracemalloc


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
        description="Swarms framework overhead with a stubbed model."
    )
    parser.add_argument("--agents", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    os.environ.setdefault("SWARMS_TELEMETRY_ON", "False")

    from swarms import Agent, ConcurrentWorkflow

    threads: list = []

    def build() -> list:
        return [
            Agent(
                agent_name=f"agent-{i}",
                llm=StubModel(args.delay, threads),
                print_on=False,
            )
            for i in range(args.agents)
        ]

    def median(call) -> float:
        return statistics.median(
            timed(call) for _ in range(args.repeat)
        )

    command = [sys.executable, "-c", "import swarms"]
    import_s = median(lambda: subprocess.run(command, check=True))
    tracemalloc.start()
    agents = build()
    agent_kb = tracemalloc.get_traced_memory()[0] / 1024 / args.agents
    workflow = ConcurrentWorkflow(agents=agents, on_error="raise")
    workflow.run("task")
    peak_mb = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    build_ms = timed(build) * 1000 / args.agents
    wall = median(lambda: workflow.run("task"))

    rows = [
        ("python -c 'import swarms' (s)", import_s),
        ("Agent construction (ms each)", build_ms),
        ("memory per Agent (KB)", agent_kb),
        (f"{args.agents}-agent fan-out wall (s)", wall),
        ("fan-out wall minus model delay (s)", wall - args.delay),
        ("fan-out peak memory, agents included (MB)", peak_mb),
        ("fan-out peak threads", max(threads)),
    ]
    print("| benchmark | swarms |\n|---|---|")
    for name, value in rows:
        print(f"| {name} | {round(value, 3)} |")


if __name__ == "__main__":
    main()
