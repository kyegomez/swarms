import multiprocessing
import signal
import time
import importlib
from typing import List
from swarms.agents.worker_system import TaskQueue, Worker


def _worker_process_main(worker_id: str, repo_path: str, db_path: str, executor_path: str):
    # Set up a graceful handler
    def _handle(sig, frame):
        raise SystemExit()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    # import executor callable
    module_name, func_name = executor_path.split(":")
    module = importlib.import_module(module_name)
    executor = getattr(module, func_name)

    tq = TaskQueue(db_path)
    w = Worker(worker_id, repo_path, tq, executor)
    try:
        w.run_loop(poll_interval=0.5)
    except SystemExit:
        return


class ProcessWorkerPool:
    """Process-based worker pool. Executors must be importable as 'module:callable'."""

    def __init__(self, repo_path: str, executor_path: str, db_path: str = ".worker_queue.db"):
        self.repo_path = repo_path
        self.executor_path = executor_path
        self.db_path = db_path
        self.processes: List[multiprocessing.Process] = []

    def spawn(self, n: int):
        for i in range(n):
            pid = f"pw-{i}"
            p = multiprocessing.Process(target=_worker_process_main, args=(pid, self.repo_path, self.db_path, self.executor_path), daemon=True)
            p.start()
            self.processes.append(p)

    def stop_all(self):
        for p in self.processes:
            try:
                p.terminate()
            except Exception:
                pass
        for p in self.processes:
            p.join(timeout=2.0)

    def add_task(self, payload: dict):
        tq = TaskQueue(self.db_path)
        return tq.add_task(payload)
