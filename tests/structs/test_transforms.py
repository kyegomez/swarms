from swarms.structs import transforms
from swarms.structs.transforms import (
    MessageTransforms,
    TransformConfig,
)
from swarms.utils import llm_backend


def test_history_inside_the_model_context_window_is_kept(monkeypatch):
    monkeypatch.setattr(
        transforms,
        "count_tokens",
        lambda text, model=None, default_encoder=None: len(
            text.split()
        ),
    )
    monkeypatch.setattr(
        llm_backend,
        "get_model_info",
        lambda model: {"max_input_tokens": 1_000_000},
        raising=False,
    )
    messages = [{"role": "system", "content": "You are helpful."}] + [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": "word " * 600,
        }
        for i in range(10)
    ]

    result = MessageTransforms(
        TransformConfig(enabled=True)
    ).transform_messages(messages, "claude-sonnet-4-6")

    assert result.was_compressed is False
    assert result.messages == messages
