import gc
import json
from unittest.mock import MagicMock
import pytest

from swarms.structs.model_router import (
    LiteLLM,
    ModelOutput,
    ModelRouter,
)


def _mock_litellm_run(monkeypatch):
    def fake_run(self, task):
        if getattr(self, "response_format", None) is ModelOutput:
            return json.dumps(
                {
                    "rationale": "test",
                    "model": "gpt-4o-mini",
                    "provider": "openai",
                    "task": task,
                    "max_tokens": 100,
                    "temperature": 0.7,
                    "system_prompt": "test prompt",
                }
            )
        return f"answer to {task}"

    monkeypatch.setattr(LiteLLM, "run", fake_run)


def test_model_router_has_executor_reuse_methods():
    router = ModelRouter()
    assert hasattr(router, "_get_executor"), "ModelRouter must have _get_executor"
    assert hasattr(router, "close"), "ModelRouter must have close()"
    assert hasattr(router, "__enter__"), "ModelRouter must support context manager protocol"
    assert hasattr(router, "__exit__"), "ModelRouter must support context manager protocol"


def test_concurrent_run_reuses_executor_across_runs(monkeypatch):
    _mock_litellm_run(monkeypatch)
    router = ModelRouter(max_workers=2)

    router.concurrent_run(["task 1", "task 2"])
    first_executor = getattr(router, "_executor", None)
    assert first_executor is not None, "Executor must be created and cached during concurrent_run"

    router.concurrent_run(["task 3", "task 4"])
    second_executor = getattr(router, "_executor", None)
    assert second_executor is first_executor, "Executor must be reused across consecutive concurrent_run calls"
    router.close()


def test_dynamic_capacity_scaling():
    router = ModelRouter(max_workers=2)

    exec1 = router._get_executor(max_workers=2)
    assert exec1._max_workers == 2

    # Smaller request reuses existing executor
    exec2 = router._get_executor(max_workers=1)
    assert exec2 is exec1

    # Larger request upscales executor
    exec3 = router._get_executor(max_workers=6)
    assert exec3 is not exec1
    assert exec3._max_workers == 6
    router.close()


def test_close_and_recreate(monkeypatch):
    _mock_litellm_run(monkeypatch)
    router = ModelRouter(max_workers=2)

    router.concurrent_run(["task 1"])
    exec1 = router._executor
    assert exec1 is not None

    router.close()
    assert router._executor is None

    router.concurrent_run(["task 2"])
    exec2 = router._executor
    assert exec2 is not None
    assert exec2 is not exec1
    router.close()


def test_context_manager(monkeypatch):
    _mock_litellm_run(monkeypatch)

    with ModelRouter(max_workers=2) as router:
        router.concurrent_run(["task 1"])
        assert router._executor is not None

    assert router._executor is None


def test_weakref_cleanup():
    router = ModelRouter(max_workers=2)
    executor = router._get_executor(max_workers=2)
    assert executor is not None

    del router
    gc.collect()
