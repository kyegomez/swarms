import gc
import weakref
from types import SimpleNamespace
import pytest

import swarms.utils.litellm_wrapper as litellm_wrapper
from swarms.structs.council_as_judge import CouncilAsAJudge


def _mock_completion(monkeypatch):
    def fake_completion(**params):
        message = SimpleNamespace(
            content="evaluation: 10/10",
            tool_calls=None,
            reasoning_content=None,
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=message, finish_reason="stop")
            ],
            usage=None,
        )

    monkeypatch.setattr(litellm_wrapper, "completion", fake_completion)


def test_council_has_executor_reuse_methods():
    council = CouncilAsAJudge(
        model_name="gpt-4o-mini",
        aggregation_model_name="gpt-4o-mini",
    )
    assert hasattr(council, "_get_executor"), "CouncilAsAJudge must have _get_executor"
    assert hasattr(council, "close"), "CouncilAsAJudge must have close()"
    assert hasattr(council, "__enter__"), "CouncilAsAJudge must support context manager"
    assert hasattr(council, "__exit__"), "CouncilAsAJudge must support context manager"


def test_council_reuses_executor_across_runs(monkeypatch):
    _mock_completion(monkeypatch)

    council = CouncilAsAJudge(
        model_name="gpt-4o-mini",
        aggregation_model_name="gpt-4o-mini",
    )

    council.run("TASK 1")
    first_executor = getattr(council, "_executor", None)
    assert first_executor is not None, "Executor should be created and retained after run()"

    council.run("TASK 2")
    second_executor = getattr(council, "_executor", None)
    assert second_executor is first_executor, "Executor must be reused across consecutive runs"
    council.close()


def test_council_get_executor_dynamic_capacity():
    council = CouncilAsAJudge(
        model_name="gpt-4o-mini",
        aggregation_model_name="gpt-4o-mini",
    )
    exec1 = council._get_executor(max_workers=2)
    assert exec1._max_workers == 2

    # Smaller or equal capacity does not recreate
    exec2 = council._get_executor(max_workers=1)
    assert exec2 is exec1

    # Larger capacity expands and replaces executor
    exec3 = council._get_executor(max_workers=6)
    assert exec3 is not exec1
    assert exec3._max_workers == 6
    council.close()


def test_council_close_and_recreate(monkeypatch):
    _mock_completion(monkeypatch)

    council = CouncilAsAJudge(
        model_name="gpt-4o-mini",
        aggregation_model_name="gpt-4o-mini",
    )
    council.run("TASK 1")
    exec1 = council._executor
    assert exec1 is not None

    council.close()
    assert council._executor is None

    council.run("TASK 2")
    exec2 = council._executor
    assert exec2 is not None
    assert exec2 is not exec1
    council.close()


def test_council_context_manager(monkeypatch):
    _mock_completion(monkeypatch)

    with CouncilAsAJudge(
        model_name="gpt-4o-mini",
        aggregation_model_name="gpt-4o-mini",
    ) as council:
        council.run("TASK 1")
        exec_ref = council._executor
        assert exec_ref is not None

    assert council._executor is None


def test_council_weakref_cleanup():
    council = CouncilAsAJudge(
        model_name="gpt-4o-mini",
        aggregation_model_name="gpt-4o-mini",
    )
    executor = council._get_executor(max_workers=2)
    assert executor is not None

    # Delete council and force garbage collection
    del council
    gc.collect()
