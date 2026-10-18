"""Exercise the real wrapper with an offline vLLM response double."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples/models/vllm/vllm_wrapper.py"
)


@pytest.fixture
def wrapper(monkeypatch: pytest.MonkeyPatch) -> object:
    def generate(
        prompts: str | list, sampling_params: object
    ) -> list:
        if isinstance(prompts, str):
            prompts = [prompts]
        return [
            SimpleNamespace(
                outputs=[SimpleNamespace(text=f"  {prompt} answer  ")]
            )
            for prompt in prompts
        ]

    engine = SimpleNamespace(generate=Mock(side_effect=generate))
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            LLM=Mock(return_value=engine),
            SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )
    monkeypatch.setitem(
        sys.modules, "loguru", SimpleNamespace(logger=Mock())
    )
    blocked = Mock(side_effect=AssertionError("No installs/network"))
    monkeypatch.setattr("subprocess.check_call", blocked)
    monkeypatch.setattr("socket.socket", blocked)
    monkeypatch.setattr("socket.create_connection", blocked)
    cls = runpy.run_path(str(EXAMPLE))["VLLMWrapper"]
    return cls(model_name="offline-test", system_prompt="System")


@pytest.mark.parametrize("batch_size", [1, 2, 10])
def test_native_batches_keep_prompts_and_response_order(
    wrapper: object, batch_size: int
) -> None:
    tasks = ["first", "second", "third", "fourth", "fifth"]
    prompts = [
        f"System\n\nUser: {task}\nAssistant:" for task in tasks
    ]
    results = wrapper.batched_run(tasks, batch_size=batch_size)
    calls = wrapper.llm.generate.call_args_list
    expected_batches = []
    for start in range(0, len(prompts), batch_size):
        stop = start + batch_size
        expected_batches.append(prompts[start:stop])
    assert [call.args[0] for call in calls] == expected_batches
    assert all(
        call.args[1] is wrapper.sampling_params for call in calls
    )
    assert results == [f"{prompt} answer" for prompt in prompts]


@pytest.mark.parametrize("batch_size", [0, -1])
def test_invalid_batch_size_does_not_generate(
    wrapper: object, batch_size: int
) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        wrapper.batched_run(["first"], batch_size=batch_size)
    wrapper.llm.generate.assert_not_called()


def test_empty_input_does_not_generate(wrapper: object) -> None:
    assert wrapper.batched_run([]) == []
    wrapper.llm.generate.assert_not_called()


def test_single_run_preserves_prompt_format(wrapper: object) -> None:
    wrapper.system_prompt = None
    assert wrapper.run("first") == "User: first\nAssistant: answer"


def test_backend_failure_propagates(wrapper: object) -> None:
    wrapper.llm.generate.side_effect = RuntimeError("offline failure")
    with pytest.raises(RuntimeError, match="offline failure"):
        wrapper.batched_run(["first"], batch_size=1)
