import threading
import time

import pytest

pytest.importorskip("mcp.server.fastmcp")

from swarms.structs.aop import (  # noqa: E402
    _agent_execution_lock,
    run_agent_isolated,
)


class FakeConversation:
    def __init__(self):
        self.messages = []


class FakeAgent:
    def __init__(self, name="worker", delay=0.0):
        self.agent_name = name
        self.short_memory = FakeConversation()
        self.delay = delay
        self.inits = 0
        self.seen = []

    def short_memory_init(self):
        self.inits += 1
        return FakeConversation()

    def run(self, task=None, **kwargs):
        self.short_memory.messages.append(task)
        if self.delay:
            time.sleep(self.delay)
        self.seen.append(list(self.short_memory.messages))
        self.last_kwargs = kwargs
        return f"answer:{task}"


class TestAgentMemoryIsolation:
    def test_the_unreset_agent_accumulates_across_calls(self):
        agent = FakeAgent()

        agent.run(task="first")
        agent.run(task="second")

        assert agent.seen[1] == ["first", "second"]

    def test_each_task_starts_from_a_fresh_conversation(self):
        agent = FakeAgent()

        run_agent_isolated(agent, task="first")
        run_agent_isolated(agent, task="second")

        assert agent.seen[0] == ["first"]
        assert agent.seen[1] == ["second"]
        assert agent.inits == 2

    def test_a_later_caller_never_sees_an_earlier_task(self):
        agent = FakeAgent()

        run_agent_isolated(agent, task="my salary is 12345")
        run_agent_isolated(agent, task="unrelated question")

        assert agent.seen[1] == ["unrelated question"]
        assert "my salary is 12345" not in agent.seen[1]

    def test_run_kwargs_are_forwarded_unchanged(self):
        agent = FakeAgent()

        run_agent_isolated(
            agent,
            task="t",
            img="a.png",
            imgs=["b.png"],
            correct_answer="42",
        )

        assert agent.last_kwargs == {
            "img": "a.png",
            "imgs": ["b.png"],
            "correct_answer": "42",
        }

    def test_the_agents_return_value_is_passed_through(self):
        agent = FakeAgent()

        assert run_agent_isolated(agent, task="t") == "answer:t"

    def test_concurrent_callers_each_see_only_their_own_task(self):
        agent = FakeAgent(delay=0.01)
        tasks = [f"task-{i}" for i in range(8)]

        threads = [
            threading.Thread(
                target=run_agent_isolated,
                args=(agent,),
                kwargs={"task": t},
            )
            for t in tasks
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(agent.seen) == len(tasks)
        assert all(len(history) == 1 for history in agent.seen)
        assert sorted(h[0] for h in agent.seen) == sorted(tasks)


class TestExecutionLock:
    def test_the_same_agent_always_gets_the_same_lock(self):
        agent = FakeAgent()

        assert _agent_execution_lock(agent) is _agent_execution_lock(
            agent
        )

    def test_separate_agents_do_not_share_a_lock(self):
        assert _agent_execution_lock(FakeAgent("a")) is not (
            _agent_execution_lock(FakeAgent("b"))
        )

    def test_two_agents_run_without_blocking_each_other(self):
        first, second = FakeAgent("a", delay=0.05), FakeAgent(
            "b", delay=0.05
        )

        started = time.time()
        threads = [
            threading.Thread(
                target=run_agent_isolated,
                args=(a,),
                kwargs={"task": "t"},
            )
            for a in (first, second)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert time.time() - started < 0.09
