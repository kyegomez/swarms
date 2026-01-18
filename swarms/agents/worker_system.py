import sqlite3
import time
import json
import threading
from typing import Optional, Callable, Any
from swarms.agents.git_utils import GitClient


class TaskQueue:
    def __init__(self, db_path: str = ".worker_queue.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    claimed_by TEXT DEFAULT NULL,
                    claimed_at REAL DEFAULT NULL,
                    result TEXT DEFAULT NULL
                )
                """
            )
            conn.commit()

    def add_task(self, payload: dict) -> int:
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO tasks (payload) VALUES (?)", (json.dumps(payload),))
            conn.commit()
            return c.lastrowid

    def claim_task(self, worker_id: str) -> Optional[dict]:
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                c = conn.cursor()
                now = time.time()
                c.execute(
                    "SELECT id FROM tasks WHERE status='pending' AND claimed_by IS NULL ORDER BY id ASC LIMIT 1"
                )
                row = c.fetchone()
                if not row:
                    return None
                task_id = row[0]
                c.execute(
                    "UPDATE tasks SET claimed_by=?, claimed_at=? WHERE id=? AND claimed_by IS NULL",
                    (worker_id, now, task_id),
                )
                if c.rowcount == 0:
                    return None
                conn.commit()
                c.execute("SELECT id, payload, status, claimed_by, claimed_at FROM tasks WHERE id=?", (task_id,))
                r = c.fetchone()
                return {"id": r[0], "payload": json.loads(r[1]), "status": r[2], "claimed_by": r[3], "claimed_at": r[4]}
        except sqlite3.OperationalError as e:
            if "no such table" in str(e).lower():
                try:
                    self._init_db()
                    with sqlite3.connect(self.db_path, timeout=10) as conn:
                        c = conn.cursor()
                        now = time.time()
                        c.execute(
                            "SELECT id FROM tasks WHERE status='pending' AND claimed_by IS NULL ORDER BY id ASC LIMIT 1"
                        )
                        row = c.fetchone()
                        if not row:
                            return None
                        task_id = row[0]
                        c.execute(
                            "UPDATE tasks SET claimed_by=?, claimed_at=? WHERE id=? AND claimed_by IS NULL",
                            (worker_id, now, task_id),
                        )
                        if c.rowcount == 0:
                            return None
                        conn.commit()
                        c.execute("SELECT id, payload, status, claimed_by, claimed_at FROM tasks WHERE id=?", (task_id,))
                        r = c.fetchone()
                        return {"id": r[0], "payload": json.loads(r[1]), "status": r[2], "claimed_by": r[3], "claimed_at": r[4]}
                except Exception:
                    return None
            raise

    def complete_task(self, task_id: int, result: dict):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("UPDATE tasks SET status='done', result=? WHERE id=?", (json.dumps(result), task_id))
            conn.commit()

    def fail_task(self, task_id: int, reason: str):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("UPDATE tasks SET status='failed', result=? WHERE id=?", (reason, task_id))
            conn.commit()

    def reclaim_stuck_tasks(self, max_age_seconds: float) -> int:
        now = time.time()
        cutoff = now - max_age_seconds
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT id FROM tasks WHERE status='pending' AND claimed_at IS NOT NULL AND claimed_at<?", (cutoff,))
            rows = c.fetchall()
            ids = [r[0] for r in rows]
            if not ids:
                return 0
            c.execute("UPDATE tasks SET claimed_by=NULL, claimed_at=NULL WHERE id IN ({seq})".format(seq=",".join(["?"] * len(ids))), ids)
            conn.commit()
            return len(ids)

    def reset_all_claims(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("UPDATE tasks SET claimed_by=NULL, claimed_at=NULL WHERE claimed_by IS NOT NULL")
            count = c.rowcount
            conn.commit()
            return count

    def stats(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM tasks")
            total = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM tasks WHERE status='done'")
            done = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM tasks WHERE status='failed'")
            failed = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM tasks WHERE status='pending' AND claimed_by IS NULL")
            pending_unclaimed = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM tasks WHERE status='pending' AND claimed_by IS NOT NULL")
            pending_claimed = c.fetchone()[0]
        return {"total": total, "done": done, "failed": failed, "pending_unclaimed": pending_unclaimed, "pending_claimed": pending_claimed}


class Worker:
    def __init__(self, worker_id: str, repo_path: str, task_queue: TaskQueue, executor: Callable[[dict, str], dict], max_retries: int = 5, timeout: int = 60 * 30, telemetry: Optional[Any] = None):
        self.worker_id = worker_id
        self.repo_path = repo_path
        self.task_queue = task_queue
        self.executor = executor
        self.max_retries = max_retries
        self.timeout = timeout
        self._stop = threading.Event()
        self.git = GitClient(repo_path)
        self.telemetry = telemetry

    def stop(self):
        self._stop.set()

    def run_loop(self, poll_interval: float = 1.0):
        while not self._stop.is_set():
            try:
                did = self.run_once()
            except Exception:
                did = False
            if not did:
                time.sleep(poll_interval)

    def run_once(self) -> bool:
        task = self.task_queue.claim_task(self.worker_id)
        if not task:
            return False
        task_id = task["id"]
        payload = task["payload"]
        if self.telemetry:
            try:
                self.telemetry.record_event("claimed", {"worker": self.worker_id, "task_id": task_id})
            except Exception:
                pass
        start = time.time()
        attempt = 0
        backoff = 1.0
        while attempt < self.max_retries and not self._stop.is_set():
            attempt += 1
            try:
                if self.telemetry:
                    try:
                        self.telemetry.record_event("execute_start", {"worker": self.worker_id, "task_id": task_id, "attempt": attempt})
                    except Exception:
                        pass
                result = self.executor(payload, self.repo_path)
                branch = result.get("branch") or f"worker/{self.worker_id}/task-{task_id}"
                message = result.get("commit_message", f"Worker {self.worker_id} task {task_id}")
                changes_dir = result.get("changes_dir")
                ok, info = self.git.commit_and_push(branch, message, changes_dir)
                if ok:
                    self.task_queue.complete_task(task_id, {"attempts": attempt, "info": info})
                    if self.telemetry:
                        try:
                            self.telemetry.record_event("complete", {"worker": self.worker_id, "task_id": task_id, "attempt": attempt})
                        except Exception:
                            pass
                    return True
                else:
                    if info.get("conflict"):
                        if self.telemetry:
                            try:
                                self.telemetry.record_event("conflict", {"worker": self.worker_id, "task_id": task_id, "attempt": attempt})
                            except Exception:
                                pass
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 60)
                        self.git.fetch()
                        self.git.rebase(branch)
                        continue
                    else:
                        self.task_queue.fail_task(task_id, json.dumps({"error": info}))
                        if self.telemetry:
                            try:
                                self.telemetry.record_event("fail", {"worker": self.worker_id, "task_id": task_id, "error": info})
                            except Exception:
                                pass
                        return False
            except Exception as e:
                last_err = str(e)
                if self.telemetry:
                    try:
                        self.telemetry.record_event("exception", {"worker": self.worker_id, "task_id": task_id, "error": last_err})
                    except Exception:
                        pass
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                if time.time() - start > self.timeout:
                    self.task_queue.fail_task(task_id, f"timeout: {last_err}")
                    if self.telemetry:
                        try:
                            self.telemetry.record_event("timeout", {"worker": self.worker_id, "task_id": task_id})
                        except Exception:
                            pass
                    return False
        self.task_queue.fail_task(task_id, "exhausted_retries")
        return False


class JudgeAgent:
    def __init__(self, task_queue: TaskQueue, repo_path: str, threshold_restart: float = 0.2):
        self.task_queue = task_queue
        self.repo_path = repo_path
        self.threshold_restart = threshold_restart

    def assess_cycle(self) -> dict:
        with sqlite3.connect(self.task_queue.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM tasks")
            total = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM tasks WHERE status='done'")
            done = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM tasks WHERE status='failed'")
            failed = c.fetchone()[0]
        progress = done / total if total else 0.0
        return {"total": total, "done": done, "failed": failed, "progress": progress}

    def decide(self) -> bool:
        stats = self.assess_cycle()
        fail_ratio = stats["failed"] / stats["total"] if stats["total"] else 0.0
        return fail_ratio < self.threshold_restart

    def fresh_start(self, git_client, reset_branch: str = "master") -> dict:
        result = {"reset": False, "claims_reset": 0, "errors": []}
        try:
            ok, info = git_client.reset_to_remote_base(reset_branch)
            result["reset"] = ok
            result["reset_info"] = info
        except Exception as e:
            result["errors"].append(str(e))
        try:
            result["claims_reset"] = self.task_queue.reset_all_claims()
        except Exception as e:
            result["errors"].append(str(e))
        return result
