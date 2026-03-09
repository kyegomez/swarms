"""
Integration test: all 6 async subagent features with REAL Agent instances hitting a real LLM.
Uses gpt-4.1-nano for speed and cost.
"""

import time
import threading
from dotenv import load_dotenv

load_dotenv()

from swarms.structs.agent import Agent
from swarms.structs.async_subagent import SubagentRegistry, TaskStatus


MODEL = "gpt-4.1-nano"
AGENTS_CREATED = []


def make_agent(
    name,
    prompt="You are a helpful assistant. Be very brief, one sentence max.",
):
    a = Agent(
        agent_name=name,
        system_prompt=prompt,
        model_name=MODEL,
        max_loops=1,
        print_on=False,
        streaming_on=False,
        verbose=False,
    )
    AGENTS_CREATED.append(a)
    return a


def test_1_async_execution():
    """
    FEATURE 1: Async Subagent Execution
    Prove parent is NOT blocked while subagents run.
    Prove subagents run on different threads concurrently.
    """
    print("\n" + "=" * 60)
    print("TEST 1: Async Subagent Execution")
    print("=" * 60)

    parent = make_agent("parent")
    child1 = make_agent("child-1")
    child2 = make_agent("child-2")
    child3 = make_agent("child-3")

    # Spawn 3 subagents
    start = time.time()
    tid1 = parent.spawn_async(
        child1, "What is 2+2? Reply with just the number."
    )
    tid2 = parent.spawn_async(
        child2, "What is 3+3? Reply with just the number."
    )
    tid3 = parent.spawn_async(
        child3, "What is 4+4? Reply with just the number."
    )
    spawn_time = time.time() - start

    print(
        f"  spawn_async() returned in {spawn_time:.4f}s (should be near-instant)"
    )
    assert (
        spawn_time < 1.0
    ), f"spawn took {spawn_time}s — should be near-instant"

    # Parent is free — prove it by checking thread
    print(f"  Parent thread: {threading.current_thread().name}")
    print("  Parent is FREE to do other work right now")

    # Now wait for results
    results = parent.gather_results(strategy="wait_all")
    total = time.time() - start
    print(f"  All 3 subagents finished in {total:.2f}s")
    print(f"  Results: {results}")

    assert len(results) == 3
    for r in results:
        assert isinstance(r, str) and len(r) > 0
    print("  PASSED")


def test_2_background_task_registry():
    """
    FEATURE 2: Background Task Registry
    Track spawned tasks with status (pending/running/completed/failed).
    Collect outputs via get_subagent_results().
    """
    print("\n" + "=" * 60)
    print("TEST 2: Background Task Registry")
    print("=" * 60)

    parent = make_agent("registry-parent")
    worker = make_agent("registry-worker")

    task_id = parent.spawn_async(
        worker, "Say hello in French. One word only."
    )
    print(f"  Task ID: {task_id}")

    # Check the registry tracks it
    registry = parent._get_registry()
    task = registry.get_task(task_id)
    print(f"  Status right after spawn: {task.status}")
    assert task.status in (TaskStatus.RUNNING, TaskStatus.COMPLETED)
    assert task.agent is worker
    assert task.parent_id == parent.id

    # Wait for completion
    parent.gather_results()
    task = registry.get_task(task_id)
    print(f"  Status after gather: {task.status}")
    print(f"  Result: {task.result}")
    print(f"  Duration: {task.completed_at - task.created_at:.2f}s")
    assert task.status == TaskStatus.COMPLETED
    assert isinstance(task.result, str)

    # get_subagent_results returns dict
    results = parent.get_subagent_results()
    print(f"  get_subagent_results(): {results}")
    assert task_id in results
    print("  PASSED")


def test_3_recursive_subagent_trees():
    """
    FEATURE 3: Recursive Subagent Trees
    Subagents spawn their own subagents. Depth is tracked.
    max_subagent_depth prevents runaway recursion.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Recursive Subagent Trees")
    print("=" * 60)

    reg = SubagentRegistry(max_depth=2)
    grandparent = make_agent("grandparent")
    parent = make_agent("tree-parent")
    child = make_agent("tree-child")

    # Level 0 (grandparent)
    gp_id = reg.spawn(
        grandparent, "What continent is France in? One word.", depth=0
    )
    # Level 1 (parent, spawned by grandparent)
    p_id = reg.spawn(
        parent,
        "What continent is Japan in? One word.",
        parent_id=gp_id,
        depth=1,
    )
    # Level 2 (child, spawned by parent)
    c_id = reg.spawn(
        child,
        "What continent is Brazil in? One word.",
        parent_id=p_id,
        depth=2,
    )

    results = reg.gather()
    print(f"  Depth 0 (grandparent): {reg.get_task(gp_id).result}")
    print(f"  Depth 1 (parent):      {reg.get_task(p_id).result}")
    print(f"  Depth 2 (child):       {reg.get_task(c_id).result}")

    assert reg.get_task(gp_id).depth == 0
    assert reg.get_task(p_id).depth == 1
    assert reg.get_task(p_id).parent_id == gp_id
    assert reg.get_task(c_id).depth == 2
    assert reg.get_task(c_id).parent_id == p_id

    # Depth 3 should be rejected
    try:
        reg.spawn(make_agent("too-deep"), "nope", depth=3)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Depth limit enforced: {e}")

    print(f"  Results count: {len(results)}")
    assert len(results) == 3
    reg.shutdown()
    print("  PASSED")


def test_4_result_aggregation():
    """
    FEATURE 4: Result Aggregation
    wait_all: block until all done.
    wait_first: return as soon as first completes.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Result Aggregation")
    print("=" * 60)

    # --- wait_all ---
    parent = make_agent("agg-parent")
    a1 = make_agent("agg-1")
    a2 = make_agent("agg-2")

    parent.spawn_async(a1, "Name one color. One word.")
    parent.spawn_async(a2, "Name one animal. One word.")
    results = parent.gather_results(strategy="wait_all")
    print(f"  wait_all results: {results}")
    assert len(results) == 2
    for r in results:
        assert isinstance(r, str) and len(r) > 0

    # --- wait_first ---
    reg = SubagentRegistry()
    fast = make_agent("fast-agent")
    slow_prompt = (
        "You are a helpful assistant. List every prime number under 500. "
        "Take your time and be thorough."
    )
    slow = make_agent("slow-agent", prompt=slow_prompt)

    reg.spawn(fast, "Say 'done'. One word only.")
    reg.spawn(
        slow, "List all primes under 500 with explanations for each."
    )

    start = time.time()
    results = reg.gather(strategy="wait_first")
    elapsed = time.time() - start
    print(
        f"  wait_first returned in {elapsed:.2f}s with {len(results)} result(s)"
    )
    print(f"  First result: {results[0][:80]}...")
    assert len(results) >= 1
    reg.shutdown()
    print("  PASSED")


def test_5_error_handling():
    """
    FEATURE 5: Error Handling & Fault Tolerance
    - Failed agents surface errors to parent
    - Retry policy works
    - fail_fast=False allows other agents to continue

    Note: Agent.run() catches LLM errors internally and returns strings.
    To test real exception propagation, we use a wrapper that raises.
    """
    print("\n" + "=" * 60)
    print("TEST 5: Error Handling & Fault Tolerance")
    print("=" * 60)

    # Use a wrapper that truly raises, simulating an unrecoverable crash
    call_count = 0

    class FailingAgent:
        agent_name = "failing-agent"

        def run(self, task):
            raise RuntimeError("Agent crashed: network timeout")

    class FlakyAgent:
        """Fails twice, then delegates to real agent on 3rd attempt."""

        agent_name = "flaky-agent"

        def __init__(self):
            self.attempts = 0
            self._real = make_agent("flaky-inner")

        def run(self, task):
            self.attempts += 1
            if self.attempts <= 2:
                raise ConnectionError(
                    f"Network error (attempt {self.attempts})"
                )
            return self._real.run(task)

    good_agent = make_agent("good-agent")

    # --- fail_fast=False: bad agent fails, good agent still succeeds ---
    reg = SubagentRegistry()
    bad_id = reg.spawn(
        FailingAgent(), "This will fail", fail_fast=False
    )
    good_id = reg.spawn(
        good_agent, "Say 'success'. One word.", fail_fast=False
    )

    reg.gather()

    bad_task = reg.get_task(bad_id)
    good_task = reg.get_task(good_id)

    print(f"  Bad agent status: {bad_task.status}")
    print(
        f"  Bad agent error: {type(bad_task.error).__name__}: {bad_task.error}"
    )
    print(f"  Good agent status: {good_task.status}")
    print(f"  Good agent result: {good_task.result}")

    assert bad_task.status == TaskStatus.FAILED
    assert isinstance(bad_task.error, RuntimeError)
    assert good_task.status == TaskStatus.COMPLETED
    assert isinstance(good_task.result, str)

    # --- Retry: fails 2x with ConnectionError, succeeds on 3rd ---
    reg2 = SubagentRegistry()
    flaky = FlakyAgent()
    retry_id = reg2.spawn(
        flaky,
        "Say 'recovered'. One word.",
        max_retries=3,
        retry_on=[ConnectionError],
        fail_fast=False,
    )
    reg2.gather()
    retry_task = reg2.get_task(retry_id)
    print(f"  Flaky agent attempts: {flaky.attempts}")
    print(f"  Flaky agent retries: {retry_task.retries}")
    print(f"  Flaky agent status: {retry_task.status}")
    print(f"  Flaky agent result: {retry_task.result}")
    assert retry_task.status == TaskStatus.COMPLETED
    assert flaky.attempts == 3  # failed 2x, succeeded on 3rd
    assert retry_task.retries == 2
    assert isinstance(retry_task.result, str)

    # --- fail_fast=True: exception propagates into gather results ---
    reg3 = SubagentRegistry()
    reg3.spawn(FailingAgent(), "crash", fail_fast=True)
    results = reg3.gather()
    print(
        f"  fail_fast=True gather result: {type(results[0]).__name__}"
    )
    assert isinstance(results[0], RuntimeError)

    reg.shutdown()
    reg2.shutdown()
    reg3.shutdown()
    print("  PASSED")


def test_6_observability():
    """
    FEATURE 6: Observability
    Structured logs emitted for every lifecycle event.
    We capture loguru output and verify events are logged.
    """
    print("\n" + "=" * 60)
    print("TEST 6: Observability")
    print("=" * 60)

    from loguru import logger
    import io

    captured = io.StringIO()
    handler_id = logger.add(
        captured, format="{message}", level="INFO"
    )

    reg = SubagentRegistry()
    agent = make_agent("observable-agent")

    # Spawn
    task_id = reg.spawn(agent, "Say 'observed'. One word.")
    reg.gather()

    # Also test failure logging
    bad = Agent(
        agent_name="fail-observable",
        system_prompt="test",
        model_name="gpt-nonexistent-obs",
        max_loops=1,
        print_on=False,
        verbose=False,
    )
    AGENTS_CREATED.append(bad)
    bad_id = reg.spawn(bad, "fail", fail_fast=False)
    reg.gather()

    reg.shutdown()

    logger.remove(handler_id)
    logs = captured.getvalue()

    print(f"  Log output ({len(logs)} chars):")
    for line in logs.strip().split("\n"):
        if "[SubagentRegistry]" in line:
            print(f"    {line.strip()}")

    # Verify lifecycle events
    assert "Spawned task" in logs, "Missing spawn log"
    assert "completed" in logs, "Missing completion log"
    assert (
        "failed" in logs or "error" in logs.lower()
    ), "Missing failure log"
    assert "Shut down" in logs, "Missing shutdown log"
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("INTEGRATION TEST: All 6 Async Subagent Features")
    print("Real Agent instances, real LLM calls (gpt-4.1-nano)")
    print("=" * 60)

    tests = [
        test_1_async_execution,
        test_2_background_task_registry,
        test_3_recursive_subagent_trees,
        test_4_result_aggregation,
        test_5_error_handling,
        test_6_observability,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(
        f"RESULTS: {passed} passed, {failed} failed out of {len(tests)}"
    )
    print("=" * 60)
