"""Quickstart example for the worker pool prototype.

Run this from the repository root. It will spawn a few workers, add tasks,
start a telemetry server at http://127.0.0.1:8008/metrics and run for a short time.
"""
from time import sleep
from swarms.agents.worker_pool import WorkerPool


def example_executor(payload, repo_path):
    # Minimal executor: no actual file changes. In real use, write files under repo_path.
    return {"branch": payload.get("branch"), "commit_message": payload.get("message", "worker change")}


if __name__ == "__main__":
    pool = WorkerPool(repo_path=".", executor=example_executor)
    pool.spawn(3)
    for i in range(10):
        pool.add_task({"message": f"auto change {i}", "branch": f"feature/auto-{i}"})
    try:
        print("Telemetry available at http://127.0.0.1:8008/metrics")
        sleep(20)
    finally:
        pool.stop_all()
