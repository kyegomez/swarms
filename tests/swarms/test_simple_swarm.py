from unittest.mock import patch
from swarms.swarms.simple_swarm import SimpleSwarm


def test_simpleswarm_initialization():
    simpleswarm = SimpleSwarm(
        num_workers=5, openai_api_key="api_key", ai_name="ai_name"
    )
    assert isinstance(simpleswarm, SimpleSwarm)
    assert len(simpleswarm.workers) == 5
    assert simpleswarm.task_queue.qsize() == 0
    assert simpleswarm.priority_queue.qsize() == 0


def test_simpleswarm_distribute():
    simpleswarm = SimpleSwarm(
        num_workers=5, openai_api_key="api_key", ai_name="ai_name"
    )
    simpleswarm.distribute("task1")
    assert simpleswarm.task_queue.qsize() == 1
    simpleswarm.distribute("task2", priority=1)
    assert simpleswarm.priority_queue.qsize() == 1


@patch("swarms.workers.worker.Worker.run")
def test_simpleswarm_process_task(mock_run):
    simpleswarm = SimpleSwarm(
        num_workers=5, openai_api_key="api_key", ai_name="ai_name"
    )
    simpleswarm._process_task("task1")
    assert mock_run.call_count == 5


def test_simpleswarm_run():
    simpleswarm = SimpleSwarm(
        num_workers=5, openai_api_key="api_key", ai_name="ai_name"
    )
    simpleswarm.distribute("task1")
    simpleswarm.distribute("task2", priority=1)
    results = simpleswarm.run()
    assert len(results) == 2


@patch("swarms.workers.Worker.run")
def test_simpleswarm_run_old(mock_run):
    simpleswarm = SimpleSwarm(
        num_workers=5, openai_api_key="api_key", ai_name="ai_name"
    )
    results = simpleswarm.run_old("task1")
    assert len(results) == 5
    assert mock_run.call_count == 5
