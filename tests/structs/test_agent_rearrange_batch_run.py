"""
Unit tests for AgentRearrange.batch_run() concurrent execution.

Tests verify:
1. batch_run uses ThreadPoolExecutor (not sequential) within each batch
2. Each task gets its own conversation copy — no shared-state corruption
3. Result ordering is preserved across all tasks and batch boundaries
4. Works correctly with and without image paths
5. batch_size boundary conditions and input validation

Note: validate_flow() requires '->' in the flow, so all pipelines here use
at least two agents.
"""

import threading
import time
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from swarms.structs.agent_rearrange import AgentRearrange

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(name: str, delay: float = 0.0):
    """Return a minimal mock Agent that sleeps *delay* seconds then echoes."""
    agent = MagicMock()
    agent.agent_name = name
    agent.system_prompt = f"I am {name}."

    def _run(task, *a, **kw):
        if delay:
            time.sleep(delay)
        return f"{name}:{task}"

    agent.run = _run
    return agent


def _make_pipeline(
    *agent_names: str, delay: float = 0.0
) -> AgentRearrange:
    """
    Build a sequential AgentRearrange pipeline with mock agents.

    Requires at least 2 agent names because validate_flow() demands '->'.
    """
    assert len(agent_names) >= 2, "_make_pipeline needs >=2 agents"
    agents = [_make_agent(n, delay=delay) for n in agent_names]
    flow = " -> ".join(agent_names)
    return AgentRearrange(
        agents=agents,
        flow=flow,
        max_loops=1,
        autosave=False,
        output_type="final",
    )


# ---------------------------------------------------------------------------
# 1. Return-value correctness
# ---------------------------------------------------------------------------


class TestBatchRunReturns:
    def test_returns_list_same_length_as_tasks(self):
        pipeline = _make_pipeline("AgentA", "AgentB")
        tasks = ["t1", "t2", "t3"]
        results = pipeline.batch_run(tasks=tasks, batch_size=10)
        assert isinstance(results, list)
        assert len(results) == len(tasks)

    def test_single_task(self):
        pipeline = _make_pipeline("AgentA", "AgentB")
        results = pipeline.batch_run(
            tasks=["only task"], batch_size=5
        )
        assert len(results) == 1

    def test_empty_task_list(self):
        pipeline = _make_pipeline("AgentA", "AgentB")
        results = pipeline.batch_run(tasks=[], batch_size=5)
        assert results == []

    def test_results_in_input_order(self):
        """Results must match the task order, not thread-completion order."""
        pipeline = _make_pipeline("AgentA", "AgentB", delay=0.02)
        tasks = [f"task-{i}" for i in range(8)]
        results = pipeline.batch_run(tasks=tasks, batch_size=8)
        for i, result in enumerate(results):
            assert (
                f"task-{i}" in result
            ), f"result[{i}] = {result!r} does not contain task-{i}"

    def test_results_across_multiple_batches_ordered(self):
        """Order must be preserved even when tasks span multiple batches."""
        pipeline = _make_pipeline("AgentA", "AgentB")
        tasks = [f"item-{i}" for i in range(7)]
        results = pipeline.batch_run(tasks=tasks, batch_size=3)
        assert len(results) == 7
        for i, result in enumerate(results):
            assert f"item-{i}" in result


# ---------------------------------------------------------------------------
# 2. Concurrency — tasks within a batch run in parallel
# ---------------------------------------------------------------------------


class TestBatchRunConcurrency:
    def test_tasks_run_concurrently(self):
        """
        Verify concurrency by asserting that at least two tasks overlap in
        execution, rather than relying on wall-clock timing thresholds that
        can be flaky under CI load.
        """
        N = 5
        pipeline = _make_pipeline("SlowAgent", "SlowAgent2")
        tasks = [f"task-{i}" for i in range(N)]

        active_count = 0
        overlap_detected = threading.Event()
        state_lock = threading.Lock()
        original_run = pipeline.run

        def instrumented_run(task, img=None, *a, **kw):
            nonlocal active_count
            with state_lock:
                active_count += 1
                if active_count >= 2:
                    overlap_detected.set()
            try:
                time.sleep(0.05)
                return original_run(task, img, *a, **kw)
            finally:
                with state_lock:
                    active_count -= 1

        pipeline.run = instrumented_run
        results = pipeline.batch_run(tasks=tasks, batch_size=N)

        assert len(results) == N
        assert (
            overlap_detected.is_set()
        ), "Expected at least two batch_run tasks to overlap in execution"

    def test_threadpoolexecutor_is_used(self):
        """Patch ThreadPoolExecutor to confirm it is invoked for each batch."""
        pipeline = _make_pipeline("AgentA", "AgentB")
        tasks = ["t1", "t2", "t3"]

        with patch(
            "swarms.structs.agent_rearrange.ThreadPoolExecutor",
            wraps=__import__(
                "concurrent.futures", fromlist=["ThreadPoolExecutor"]
            ).ThreadPoolExecutor,
        ) as mock_tpe:
            pipeline.batch_run(tasks=tasks, batch_size=10)
            assert mock_tpe.call_count == 1

    def test_multiple_batches_uses_executor_per_batch(self):
        """One ThreadPoolExecutor context-manager per batch."""
        pipeline = _make_pipeline("AgentA", "AgentB")
        tasks = [f"t{i}" for i in range(6)]

        with patch(
            "swarms.structs.agent_rearrange.ThreadPoolExecutor",
            wraps=__import__(
                "concurrent.futures", fromlist=["ThreadPoolExecutor"]
            ).ThreadPoolExecutor,
        ) as mock_tpe:
            pipeline.batch_run(tasks=tasks, batch_size=2)
            # 6 tasks / batch_size=2 -> 3 batches -> 3 executor instances
            assert mock_tpe.call_count == 3


# ---------------------------------------------------------------------------
# 3. Conversation isolation — no shared state between tasks
# ---------------------------------------------------------------------------


class TestConversationIsolation:
    def test_original_conversation_not_mutated(self):
        """The pipeline's own conversation should not change after batch_run."""
        pipeline = _make_pipeline("AgentA", "AgentB")
        original_msg_count = len(
            pipeline.conversation.conversation_history
        )

        pipeline.batch_run(tasks=["task1", "task2"], batch_size=5)

        after_msg_count = len(
            pipeline.conversation.conversation_history
        )
        assert (
            after_msg_count == original_msg_count
        ), "pipeline.conversation was mutated by batch_run"

    def test_each_task_gets_own_conversation_copy(self):
        """Each worker clone must have its own conversation, not share one."""
        pipeline = _make_pipeline("AgentA", "AgentB")
        tasks = ["alpha", "beta", "gamma"]

        # Hold strong references so GC doesn't reuse addresses
        seen_conversations: list = []

        def _mock_run(self_inner, task=None, img=None, *a, **kw):
            seen_conversations.append(self_inner.conversation)
            return f"result:{task}"

        with patch.object(AgentRearrange, "_run", _mock_run):
            pipeline.batch_run(tasks=tasks, batch_size=10)

        assert len(seen_conversations) == len(tasks)
        # All conversation objects must be distinct instances
        for i in range(len(seen_conversations)):
            for j in range(i + 1, len(seen_conversations)):
                assert (
                    seen_conversations[i] is not seen_conversations[j]
                ), f"Tasks {i} and {j} shared the same conversation object"

    def test_stateful_agent_state_does_not_bleed_across_tasks(self):
        """
        Regression: concurrent tasks must not corrupt each other's results
        via shared agent state. With deepcopy each task owns its own agent
        instance, so last_task cannot be overwritten by a racing thread.
        """

        class StatefulAgent:
            def __init__(self, name: str):
                self.agent_name = name
                self.system_prompt = ""
                self.last_task = None

            def run(self, task, *args, **kwargs):
                time.sleep(0.05)
                self.last_task = task
                return f"{self.agent_name}::{self.last_task}"

        pipeline = AgentRearrange(
            agents=[StatefulAgent("A"), StatefulAgent("B")],
            flow="A -> B",
            max_loops=1,
            autosave=False,
            output_type="final",
        )

        N = 10
        tasks = [f"t{i}" for i in range(N)]
        results = pipeline.batch_run(tasks=tasks, batch_size=N)

        assert len(results) == N
        for i, result in enumerate(results):
            assert f"t{i}" in str(
                result
            ), f"result[{i}] missing 't{i}': {result!r} — agent state race detected"


# ---------------------------------------------------------------------------
# 4. Image paths forwarded correctly
# ---------------------------------------------------------------------------


class TestBatchRunWithImages:
    def test_img_list_passed_per_task(self):
        """When img is provided, each task receives its corresponding image path."""
        received: List[Optional[str]] = []

        agent_a = _make_agent("AgentA")
        agent_b = _make_agent("AgentB")

        pipeline = AgentRearrange(
            agents=[agent_a, agent_b],
            flow="AgentA -> AgentB",
            max_loops=1,
            autosave=False,
            output_type="final",
        )

        tasks = ["t1", "t2", "t3"]
        images = ["img1.png", "img2.png", "img3.png"]

        def _mock_run(self_inner, task=None, img=None, *a, **kw):
            received.append(img)
            return f"result:{task}"

        with patch.object(AgentRearrange, "_run", _mock_run):
            results = pipeline.batch_run(
                tasks=tasks, img=images, batch_size=5
            )

        assert len(results) == 3
        assert sorted(received) == sorted(images)

    def test_no_img_passes_none(self):
        """When img is omitted, None is passed as img for every task."""
        received_imgs: List[Optional[str]] = []

        agent_a = _make_agent("AgentA")
        agent_b = _make_agent("AgentB")

        pipeline = AgentRearrange(
            agents=[agent_a, agent_b],
            flow="AgentA -> AgentB",
            max_loops=1,
            autosave=False,
            output_type="final",
        )

        def _mock_run(self_inner, task=None, img=None, *a, **kw):
            received_imgs.append(img)
            return f"result:{task}"

        with patch.object(AgentRearrange, "_run", _mock_run):
            pipeline.batch_run(tasks=["t1", "t2"], batch_size=5)

        assert all(img is None for img in received_imgs)


# ---------------------------------------------------------------------------
# 5. batch_size boundary conditions and input validation
# ---------------------------------------------------------------------------


class TestBatchSizeBoundaries:
    @pytest.mark.parametrize("batch_size", [1, 2, 3, 5, 100])
    def test_various_batch_sizes_return_all_results(self, batch_size):
        pipeline = _make_pipeline("AgentA", "AgentB")
        tasks = [f"task-{i}" for i in range(5)]
        results = pipeline.batch_run(
            tasks=tasks, batch_size=batch_size
        )
        assert len(results) == len(tasks)

    def test_batch_size_one_still_uses_executor(self):
        """Even batch_size=1 should go through ThreadPoolExecutor."""
        pipeline = _make_pipeline("AgentA", "AgentB")
        with patch(
            "swarms.structs.agent_rearrange.ThreadPoolExecutor",
            wraps=__import__(
                "concurrent.futures", fromlist=["ThreadPoolExecutor"]
            ).ThreadPoolExecutor,
        ) as mock_tpe:
            pipeline.batch_run(tasks=["only"], batch_size=1)
            assert mock_tpe.call_count == 1

    @pytest.mark.parametrize("batch_size", [0, -1, -10])
    def test_invalid_batch_size_raises(self, batch_size):
        """batch_size <= 0 must raise ValueError immediately."""
        pipeline = _make_pipeline("AgentA", "AgentB")
        with pytest.raises(ValueError, match="batch_size"):
            pipeline.batch_run(tasks=["t1"], batch_size=batch_size)

    def test_img_length_mismatch_raises(self):
        """Mismatched img and tasks lengths must raise ValueError."""
        pipeline = _make_pipeline("AgentA", "AgentB")
        with pytest.raises(ValueError, match="img length"):
            pipeline.batch_run(
                tasks=["t1", "t2"], img=["img1.png"], batch_size=5
            )
