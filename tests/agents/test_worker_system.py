import os
import time
import sqlite3
from swarms.agents.worker_system import TaskQueue, Worker


def test_taskqueue_add_claim_complete(tmp_path):
    db = tmp_path / "queue.db"
    tq = TaskQueue(str(db))
    tid = tq.add_task({"foo": "bar"})
    assert isinstance(tid, int) and tid > 0

    claim = tq.claim_task("worker-1")
    assert claim is not None
    assert claim["id"] == tid

    tq.complete_task(tid, {"ok": True})
    stats = tq.stats()
    assert stats["done"] == 1


def test_reclaim_stuck_tasks(tmp_path):
    db = tmp_path / "queue2.db"
    tq = TaskQueue(str(db))
    tid = tq.add_task({"x": 1})
    # claim via direct DB update to simulate old claimed_at
    with sqlite3.connect(tq.db_path) as conn:
        c = conn.cursor()
        now = time.time() - 10000
        c.execute("UPDATE tasks SET claimed_by=?, claimed_at=? WHERE id=?", ("w", now, tid))
        conn.commit()

    reclaimed = tq.reclaim_stuck_tasks(60)
    assert reclaimed >= 1
    stats = tq.stats()
    assert stats["pending_unclaimed"] >= 1


class DummyGit:
    def __init__(self, repo_path):
        self.repo_path = repo_path

    def commit_and_push(self, branch, message, changes_dir=None):
        return True, {"out": "ok"}

    def fetch(self):
        return 0, "fetched"

    def rebase(self, branch):
        return 0, "rebased"


def test_worker_run_once_success(tmp_path, monkeypatch):
    db = tmp_path / "queue3.db"
    tq = TaskQueue(str(db))
    tid = tq.add_task({"task": "do"})

    # create worker with dummy executor and dummy git
    def executor(payload, repo_path):
        return {"branch": "feature/test", "commit_message": "msg"}

    w = Worker("w-test", ".", tq, executor, max_retries=2, timeout=5)
    # inject dummy git client
    w.git = DummyGit('.')

    result = w.run_once()
    assert result is True
    stats = tq.stats()
    assert stats["done"] == 1
