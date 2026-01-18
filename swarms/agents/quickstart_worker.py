import tempfile
import time
from swarms.agents.process_worker_pool import ProcessWorkerPool
from swarms.agents.worker_system import TaskQueue


def main():
    tmp = tempfile.TemporaryDirectory()
    db_path = tmp.name + "/quickstart.db"

    pool = ProcessWorkerPool(repo_path='.', executor_path='swarms.agents.simple_executors:noop_executor', db_path=db_path)

    # add tasks
    for i in range(6):
        pool.add_task({"message": f"task-{i}", "branch": f"feature/quick-{i}"})

    print("Spawning 2 worker processes...")
    pool.spawn(2)

    # wait for workers to process tasks
    time.sleep(3)

    tq = TaskQueue(db_path)
    stats = tq.stats()
    print("Task stats:", stats)

    pool.stop_all()
    tmp.cleanup()


if __name__ == '__main__':
    main()
