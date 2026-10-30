import concurrent.futures
import pytest
from swarms.structs.multi_agent_exec import (
    run_agents_concurrently,
    batched_grid_agent_execution,
)


class MockAgent:
    def __init__(self, name: str, return_value: str = "ok"):
        self.agent_name = name
        self.return_value = return_value
        self.calls = []

    def run(self, task: str = None, img: str = None, *args, **kwargs):
        self.calls.append((task, img))
        return f"{self.agent_name}:{self.return_value}:{task}"


class FailingAgent:
    def __init__(self, name: str):
        self.agent_name = name

    def run(self, task: str = None, img: str = None, *args, **kwargs):
        raise RuntimeError(f"{self.agent_name} boom")


def test_run_agents_concurrently_with_custom_executor():
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    try:
        a1 = MockAgent("Agent1")
        a2 = MockAgent("Agent2")
        results = run_agents_concurrently(
            agents=[a1, a2],
            task="do_work",
            executor=pool,
        )
        assert results == ["Agent1:ok:do_work", "Agent2:ok:do_work"]
        # Executor must remain active and functional
        f = pool.submit(lambda: 123)
        assert f.result() == 123
    finally:
        pool.shutdown(wait=True)


def test_run_agents_concurrently_dict_output_with_custom_executor():
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    try:
        a1 = MockAgent("Agent1")
        a2 = MockAgent("Agent2")
        results = run_agents_concurrently(
            agents=[a1, a2],
            task="do_work",
            return_agent_output_dict=True,
            executor=pool,
        )
        assert results == {
            "Agent1": "Agent1:ok:do_work",
            "Agent2": "Agent2:ok:do_work",
        }
        # Executor must remain active
        f = pool.submit(lambda: 456)
        assert f.result() == 456
    finally:
        pool.shutdown(wait=True)


def test_run_agents_concurrently_reuses_executor_across_calls():
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    try:
        a1 = MockAgent("A1")
        a2 = MockAgent("A2")
        res1 = run_agents_concurrently(
            agents=[a1, a2], task="task1", executor=pool
        )
        assert res1 == ["A1:ok:task1", "A2:ok:task1"]

        res2 = run_agents_concurrently(
            agents=[a1, a2], task="task2", executor=pool
        )
        assert res2 == ["A1:ok:task2", "A2:ok:task2"]
    finally:
        pool.shutdown(wait=True)


def test_run_agents_concurrently_with_executor_handles_failure():
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    try:
        a1 = MockAgent("A1")
        a2 = FailingAgent("A2")
        results = run_agents_concurrently(
            agents=[a1, a2], task="task1", executor=pool
        )
        assert results[0] == "A1:ok:task1"
        assert isinstance(results[1], RuntimeError)
        # Executor is still alive
        f = pool.submit(lambda: 789)
        assert f.result() == 789
    finally:
        pool.shutdown(wait=True)


def test_batched_grid_agent_execution_with_custom_executor():
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    try:
        a1 = MockAgent("Agent1")
        a2 = MockAgent("Agent2")
        results = batched_grid_agent_execution(
            agents=[a1, a2],
            tasks=["task_a", "task_b"],
            executor=pool,
        )
        assert results == ["Agent1:ok:task_a", "Agent2:ok:task_b"]
        # Executor must remain active
        f = pool.submit(lambda: 101)
        assert f.result() == 101
    finally:
        pool.shutdown(wait=True)


def test_batched_grid_agent_execution_reuses_executor_across_calls():
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    try:
        a1 = MockAgent("A1")
        a2 = MockAgent("A2")
        res1 = batched_grid_agent_execution(
            agents=[a1, a2], tasks=["t1", "t2"], executor=pool
        )
        assert res1 == ["A1:ok:t1", "A2:ok:t2"]

        res2 = batched_grid_agent_execution(
            agents=[a1, a2], tasks=["t3", "t4"], executor=pool
        )
        assert res2 == ["A1:ok:t3", "A2:ok:t4"]
    finally:
        pool.shutdown(wait=True)


def test_batched_grid_agent_execution_with_executor_handles_failure():
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    try:
        a1 = MockAgent("A1")
        a2 = FailingAgent("A2")
        results = batched_grid_agent_execution(
            agents=[a1, a2], tasks=["t1", "t2"], executor=pool
        )
        assert results[0] == "A1:ok:t1"
        assert isinstance(results[1], RuntimeError)
        # Executor must remain active
        f = pool.submit(lambda: 202)
        assert f.result() == 202
    finally:
        pool.shutdown(wait=True)
