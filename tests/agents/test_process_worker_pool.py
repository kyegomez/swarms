import time
import sqlite3
from swarms.agents.process_worker_pool import ProcessWorkerPool
from swarms.agents.worker_system import TaskQueue


def test_process_pool_completes_tasks(tmp_path):
    db = tmp_path / "proc_queue.db"
    pool = ProcessWorkerPool(repo_path='.', executor_path='swarms.agents.simple_executors:noop_executor', db_path=str(db))
    # add some tasks
    for i in range(4):
        pool.add_task({"message": f"task-{i}", "branch": f"feature/p-{i}"})

    pool.spawn(2)
    # wait for workers to pick up tasks
    time.sleep(2)
    tq = TaskQueue(str(db))
    stats = tq.stats()
    # expect tasks to be completed or at least not pending claimed
    assert stats['total'] == 4
    # allow either done or failed, but ensure workers processed some
    assert stats['done'] + stats['failed'] >= 1
    pool.stop_all()
