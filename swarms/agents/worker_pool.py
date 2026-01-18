import threading
from typing import List, Callable
from swarms.agents.worker_system import Worker, TaskQueue, JudgeAgent
from swarms.agents.observability import TelemetryCollector, TelemetryServer
from swarms.agents.git_utils import GitClient
import time


class WorkerPool:
    """Thread-based worker pool for running multiple Worker instances concurrently.

    Note: This is a prototype using threads. For heavy isolation and CPU scaling,
    replace with multiprocessing or orchestrate workers as separate processes.
    """

    def __init__(self, repo_path: str, executor: Callable[[dict, str], dict]):
        self.repo_path = repo_path
        self.executor = executor
        self.task_queue = TaskQueue()
        self.workers: List[Worker] = []
        self.threads: List[threading.Thread] = []
        self.telemetry = TelemetryCollector()
        self.telemetry_server = TelemetryServer(self.telemetry)
        self._monitor_thread = None
        self._monitor_stop = threading.Event()
        self.git = GitClient(repo_path)
        self.judge = JudgeAgent(self.task_queue, repo_path)

    def spawn(self, n: int):
        # start telemetry server and monitor on first spawn
        if not self._monitor_thread:
            try:
                self.telemetry_server.start()
            except Exception:
                pass
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
        for i in range(n):
            wid = f"w-{i}"
            w = Worker(wid, self.repo_path, self.task_queue, self.executor, telemetry=self.telemetry)
            t = threading.Thread(target=w.run_loop, daemon=True)
            t.start()
            self.workers.append(w)
            self.threads.append(t)

    def stop_all(self):
        for w in self.workers:
            w.stop()
        for t in self.threads:
            t.join(timeout=1.0)
        self._monitor_stop.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        try:
            self.telemetry_server.stop()
        except Exception:
            pass

    def add_task(self, payload: dict):
        return self.task_queue.add_task(payload)

    def _monitor_loop(self, reclaim_age: float = 60 * 5, poll: float = 5.0):
        # reclaim tasks claimed longer than reclaim_age and report metrics
        while not self._monitor_stop.is_set():
            reclaimed = self.task_queue.reclaim_stuck_tasks(reclaim_age)
            if reclaimed and self.telemetry:
                self.telemetry.record_event("reclaimed", {"count": reclaimed})
            # emit periodic stats
            stats = self.task_queue.stats()
            if self.telemetry:
                self.telemetry.record_event("stats", stats)
            # check judge decision and possibly trigger fresh start
            try:
                ok = self.judge.decide()
                if not ok:
                    info = self.judge.fresh_start(self.git, reset_branch="master")
                    if self.telemetry:
                        self.telemetry.record_event("fresh_start", info)
            except Exception as e:
                if self.telemetry:
                    self.telemetry.record_event("judge_error", {"error": str(e)})

            # respawn dead threads/workers
            for idx, t in enumerate(list(self.threads)):
                if not t.is_alive():
                    # attempt to restart corresponding worker
                    try:
                        old_w = self.workers[idx]
                        new_wid = f"{old_w.worker_id}-r"
                        new_w = Worker(new_wid, self.repo_path, self.task_queue, self.executor, telemetry=self.telemetry)
                        new_t = threading.Thread(target=new_w.run_loop, daemon=True)
                        new_t.start()
                        self.workers[idx] = new_w
                        self.threads[idx] = new_t
                        if self.telemetry:
                            self.telemetry.record_event("respawn", {"old": old_w.worker_id, "new": new_wid})
                    except Exception as e:
                        if self.telemetry:
                            self.telemetry.record_event("respawn_error", {"error": str(e)})
            time.sleep(poll)
