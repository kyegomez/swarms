import pytest

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


system = {"role": "System", "content": "You are helpful."}
memory = {"role": "System", "content": "Project: Helios."}
summary = {"role": "System", "content": "Earlier: Helios."}
turns = [
    {"role": role, "content": f"{role} {i}"}
    for i in range(3)
    for role in ("User", "Agent")
]


@pytest.mark.parametrize(
    "config, messages, expected",
    [
        (
            TransformConfig(
                enabled=True,
                max_messages=5,
                preserve_recent_messages=2,
            ),
            [system, memory, *turns],
            [system, memory, turns[2], *turns[-2:]],
        ),
        (
            TransformConfig(enabled=True, max_tokens=10),
            [system, memory, summary, *turns[:2]],
            [system, memory, summary, turns[1]],
        ),
    ],
)
def test_compression_keeps_conversation_system_rows(
    monkeypatch, config, messages, expected
):
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

    result = MessageTransforms(config).transform_messages(messages)

    assert result.messages == expected


def test_long_history_just_over_max_tokens_loses_one_message(
    monkeypatch,
):
    monkeypatch.setattr(
        transforms,
        "count_tokens",
        lambda text, model=None, default_encoder=None: len(
            text.split()
        ),
    )
    messages = [
        {"role": "user", "content": f"turn {i}"} for i in range(60)
    ]

    result = MessageTransforms(
        TransformConfig(enabled=True, max_tokens=119)
    ).transform_messages(messages)

    assert result.compressed_message_count == 59
    assert result.messages[0] == messages[0]
    assert result.messages[-1] == messages[-1]
