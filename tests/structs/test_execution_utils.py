"""
Tests for ``swarms.structs.execution_utils.batched_run``.

Offline: the callable under test is always a local function, never an LLM.
The defaults here are a contract - twelve structures replaced a hand-written
``[self.run(task) for task in tasks]`` with this, so a list in task order with
exceptions propagating is what they are relying on.
"""

import pytest

from swarms.structs.execution_utils import (
    batched_run,
    run_concurrently,
)


def echo(task, *args, **kwargs):
    return {"task": task, "args": args, "kwargs": kwargs}


def plain(task):
    """No ``img`` parameter - the common ``run(self, task)`` shape."""
    return f"<{task}>"


def boom(task):
    raise ValueError(f"kaboom: {task}")


class TestDefaults:
    """What the twelve migrated methods depend on."""

    def test_returns_a_list_not_a_dict(self):
        assert batched_run(plain, ["a", "b"]) == ["<a>", "<b>"]

    def test_order_is_task_order(self):
        tasks = [str(i) for i in range(25)]
        assert batched_run(plain, tasks) == [f"<{t}>" for t in tasks]

    def test_duplicate_tasks_are_not_collapsed(self):
        """A dict-keyed result would silently drop the second one."""
        assert batched_run(plain, ["a", "a"]) == ["<a>", "<a>"]

    def test_exceptions_propagate(self):
        with pytest.raises(ValueError, match="kaboom: x"):
            batched_run(boom, ["x"])

    def test_callable_without_img_still_works(self):
        """``img`` must not be forwarded when it was never set."""
        assert batched_run(plain, ["a"]) == ["<a>"]

    def test_empty_tasks(self):
        assert batched_run(plain, []) == []
        assert (
            batched_run(plain, [], return_agent_output_dict=True)
            == {}
        )

    def test_results_are_not_stringified(self):
        out = batched_run(echo, ["a"])
        assert isinstance(out[0], dict)


class TestForwarding:
    def test_args_and_kwargs_reach_the_callable(self):
        out = batched_run(echo, ["t"], 1, 2, flag=True)
        assert out[0]["args"] == (1, 2)
        assert out[0]["kwargs"] == {"flag": True}

    def test_img_is_passed_as_a_keyword(self):
        out = batched_run(echo, ["t"], img="x.png")
        assert out[0]["kwargs"]["img"] == "x.png"

    def test_img_is_absent_when_none(self):
        assert "img" not in batched_run(echo, ["t"])[0]["kwargs"]


class TestImgs:
    """``imgs`` pairs one image per task, by position."""

    def test_paired_by_position(self):
        out = batched_run(echo, ["a", "b"], imgs=["1.png", "2.png"])
        assert out[0]["kwargs"]["img"] == "1.png"
        assert out[1]["kwargs"]["img"] == "2.png"

    def test_pairing_holds_when_concurrent(self):
        out = batched_run(
            echo, ["a", "b"], imgs=["1.png", "2.png"], max_workers=2
        )
        assert [o["kwargs"]["img"] for o in out] == [
            "1.png",
            "2.png",
        ]

    def test_wrong_length_raises_rather_than_truncating(self):
        """zip() would silently run one task instead of two."""
        with pytest.raises(ValueError, match="one image per task"):
            batched_run(echo, ["a", "b"], imgs=["only.png"])

    def test_a_none_entry_omits_img_for_that_task(self):
        out = batched_run(echo, ["a", "b"], imgs=[None, "2.png"])
        assert "img" not in out[0]["kwargs"]
        assert out[1]["kwargs"]["img"] == "2.png"

    def test_a_sequence_in_img_is_treated_as_per_task_images(self):
        """Callers spell this parameter both ways; broadcasting a list would
        hand every task the whole list as its image."""
        out = batched_run(echo, ["a", "b"], img=["1.png", "2.png"])
        assert out[0]["kwargs"]["img"] == "1.png"
        assert out[1]["kwargs"]["img"] == "2.png"

    def test_a_sequence_in_img_is_length_checked(self):
        with pytest.raises(ValueError, match="one image per task"):
            batched_run(echo, ["a", "b", "c"], img=["only.png"])

    def test_a_sequence_in_img_pairs_when_concurrent(self):
        out = batched_run(
            echo, ["a", "b"], img=["1.png", "2.png"], max_workers=2
        )
        assert [o["kwargs"]["img"] for o in out] == [
            "1.png",
            "2.png",
        ]

    def test_a_string_img_is_never_iterated_per_character(self):
        """zip(tasks, "x.png") paired task 0 with 'x' and task 1 with '.'."""
        out = batched_run(echo, ["a", "b"], img="x.png")
        assert all(o["kwargs"]["img"] == "x.png" for o in out)

    def test_img_and_imgs_together_is_rejected(self):
        with pytest.raises(ValueError, match="not both"):
            batched_run(echo, ["a"], img="x.png", imgs=["y.png"])

    def test_img_broadcasts_to_every_task(self):
        out = batched_run(echo, ["a", "b"], img="same.png")
        assert all(o["kwargs"]["img"] == "same.png" for o in out)


class TestConcurrency:
    def test_order_survives_out_of_order_completion(self):
        """The slowest task is first; it must still come back first."""
        import time

        def slow_first(task):
            time.sleep(0.05 if task == "0" else 0)
            return task

        tasks = [str(i) for i in range(6)]
        assert batched_run(slow_first, tasks, max_workers=6) == tasks

    def test_concurrent_matches_sequential(self):
        tasks = [str(i) for i in range(12)]
        assert batched_run(plain, tasks) == batched_run(
            plain, tasks, max_workers=4
        )

    def test_exceptions_propagate_when_concurrent(self):
        with pytest.raises(ValueError):
            batched_run(boom, ["x"], max_workers=2)


class TestOptions:
    def test_dict_mode(self):
        assert batched_run(
            plain, ["a", "b"], return_agent_output_dict=True
        ) == {"a": "<a>", "b": "<b>"}

    def test_return_exceptions_captures_instead_of_raising(self):
        out = batched_run(boom, ["x", "y"], return_exceptions=True)
        assert all(isinstance(r, ValueError) for r in out)
        assert len(out) == 2

    def test_return_exceptions_keeps_good_results(self):
        def half(task):
            if task == "bad":
                raise ValueError("no")
            return task

        out = batched_run(half, ["ok", "bad"], return_exceptions=True)
        assert out[0] == "ok"
        assert isinstance(out[1], ValueError)


class TestValidation:
    def test_non_callable_raises_typeerror(self):
        with pytest.raises(TypeError, match="must be callable"):
            batched_run("not callable", ["a"])

    @pytest.mark.parametrize("workers", [0, -1])
    def test_bad_max_workers_raises(self, workers):
        with pytest.raises(ValueError, match="max_workers"):
            batched_run(plain, ["a"], max_workers=workers)

    def test_a_generator_of_tasks_is_accepted(self):
        assert batched_run(plain, (t for t in "ab")) == [
            "<a>",
            "<b>",
        ]


class TestRunConcurrently:
    """``run_concurrently`` is ``batched_run`` with concurrency on by default."""

    def test_returns_a_list_in_task_order(self):
        assert run_concurrently(plain, ["a", "b", "c"]) == [
            "<a>",
            "<b>",
            "<c>",
        ]

    def test_order_survives_out_of_order_completion(self):
        import time

        def slow_first(task):
            time.sleep(0.05 if task == "0" else 0)
            return task

        tasks = [str(i) for i in range(6)]
        assert run_concurrently(slow_first, tasks) == tasks

    def test_it_actually_runs_in_parallel(self):
        """Sequential would take 6x as long."""
        import time

        def slow(task):
            time.sleep(0.05)
            return task

        start = time.time()
        run_concurrently(slow, list("abcdef"), max_workers=6)
        assert time.time() - start < 0.2

    def test_matches_batched_run_results(self):
        tasks = [str(i) for i in range(10)]
        assert run_concurrently(plain, tasks) == batched_run(
            plain, tasks
        )

    def test_args_and_kwargs_reach_the_callable(self):
        out = run_concurrently(echo, ["t"], 1, flag=True)
        assert out[0]["args"] == (1,)
        assert out[0]["kwargs"] == {"flag": True}

    def test_exceptions_propagate(self):
        with pytest.raises(ValueError):
            run_concurrently(boom, ["x"])

    def test_options_pass_through(self):
        assert run_concurrently(
            plain, ["a"], return_agent_output_dict=True
        ) == {"a": "<a>"}
        out = run_concurrently(boom, ["x"], return_exceptions=True)
        assert isinstance(out[0], ValueError)

    def test_empty_tasks(self):
        assert run_concurrently(plain, []) == []


class TestTracingContext:
    def test_concurrent_path_uses_the_context_executor(self):
        """A plain ThreadPoolExecutor orphans worker spans from the run."""
        import swarms.structs.execution_utils as eu
        from swarms.telemetry.otel import ContextThreadPoolExecutor

        assert (
            eu.ContextThreadPoolExecutor is ContextThreadPoolExecutor
        )
        assert issubclass(
            ContextThreadPoolExecutor,
            __import__(
                "concurrent.futures", fromlist=["ThreadPoolExecutor"]
            ).ThreadPoolExecutor,
        )

    def test_context_executor_is_the_one_actually_used(
        self, monkeypatch
    ):
        import swarms.structs.execution_utils as eu

        used = []
        real = eu.ContextThreadPoolExecutor

        class Spy(real):
            def __init__(self, *a, **kw):
                used.append(True)
                super().__init__(*a, **kw)

        monkeypatch.setattr(eu, "ContextThreadPoolExecutor", Spy)
        run_concurrently(plain, ["a", "b"], max_workers=2)
        assert (
            used
        ), "concurrent path did not use the context executor"


class TestMigratedCallers:
    """The behaviours the migrated methods depend on, pinned here.

    Each of these replaced a hand-written loop that got one of them wrong.
    """

    def test_index_alignment_survives_failures(self):
        """HybridHierarchicalClusterSwarm used as_completed, so a slow task
        pushed later results into earlier slots."""
        import time

        def flaky(task):
            if task == "slow":
                time.sleep(0.05)
                return "slow-done"
            if task == "bad":
                raise RuntimeError("boom")
            return task

        out = batched_run(
            flaky,
            ["slow", "bad", "c"],
            max_workers=3,
            return_exceptions=True,
        )
        assert out[0] == "slow-done"
        assert isinstance(out[1], RuntimeError)
        assert out[2] == "c"

    def test_chunked_and_pooled_give_the_same_order(self):
        """LiteLLM ran fixed-size chunks; a single pool must match it."""
        tasks = [str(i) for i in range(20)]
        assert batched_run(plain, tasks, max_workers=4) == [
            f"<{t}>" for t in tasks
        ]

    def test_keyword_task_callers_still_bind_correctly(self):
        """AdvisorSwarm called run(task=t, *args), which raised whenever
        args was non-empty."""

        def run(task=None, img=None):
            return (task, img)

        assert batched_run(run, ["t"], "positional.png") == [
            ("t", "positional.png")
        ]
