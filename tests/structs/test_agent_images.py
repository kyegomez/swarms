"""Images given to an Agent reach the model request."""

import asyncio
import base64
import struct
import zlib
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

import swarms.utils.litellm_wrapper as litellm_wrapper
from swarms import Agent
from swarms.schemas.agent_errors import AgentLLMError


def _png(size: int = 16) -> bytes:
    """Build a small valid PNG, long enough to be read as raw base64."""
    rows = b"".join(
        b"\x00" + bytes((x * y) % 256 for x in range(size * 3))
        for y in range(size)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return (
            struct.pack(">I", len(data))
            + body
            + struct.pack(">I", zlib.crc32(body))
        )

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


PNG_BYTES = _png()
RAW_BASE64 = base64.b64encode(PNG_BYTES).decode()
DATA_URI = f"data:image/png;base64,{RAW_BASE64}"
URL = "https://example.com/chart.png"
TASK = "What is in this image?"


def _reply(*args: Any, **kwargs: Any) -> SimpleNamespace:
    message = SimpleNamespace(content="ok", tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)], usage=None
    )


async def _async_reply(*args: Any, **kwargs: Any) -> SimpleNamespace:
    return _reply()


def _agent(model_name: str = "gpt-4.1", **kwargs: Any) -> Agent:
    kwargs.setdefault("max_loops", 1)
    return Agent(
        agent_name="Vision",
        model_name=model_name,
        print_on=False,
        persistent_memory=False,
        **kwargs,
    )


def _requests(run: Any) -> List[List[Dict[str, Any]]]:
    """Run with the provider stubbed and return each request's messages."""
    with patch.object(
        litellm_wrapper, "completion", side_effect=_reply
    ) as completion, patch.object(
        litellm_wrapper, "acompletion", side_effect=_async_reply
    ) as acompletion:
        run()

    calls = completion.call_args_list + acompletion.call_args_list
    return [call.kwargs["messages"] for call in calls]


def _image_urls(messages: List[Dict[str, Any]]) -> List[str]:
    return [
        part["image_url"]["url"]
        for message in messages
        if isinstance(message["content"], list)
        for part in message["content"]
        if part["type"] == "image_url"
    ]


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


@pytest.mark.parametrize(
    "run_kwargs, agent_kwargs, expected",
    [
        ({"img": URL}, {}, [URL]),
        ({"imgs": [URL, DATA_URI]}, {}, [URL, DATA_URI]),
        ({"img": URL, "imgs": [DATA_URI]}, {}, [URL, DATA_URI]),
        ({"img": URL}, {"system_prompt": "Be brief."}, [URL]),
        ({"img": URL}, {"model_name": "claude-sonnet-4-6"}, [URL]),
        ({}, {}, []),
    ],
)
def test_run_sends_the_images_with_the_task(
    run_kwargs, agent_kwargs, expected
):
    agent = _agent(**agent_kwargs)

    (messages,) = _requests(lambda: agent.run(TASK, **run_kwargs))

    assert _image_urls(messages) == expected
    assert messages[-1]["role"] == "user"
    assert TASK in str(messages[-1]["content"])


@pytest.mark.parametrize("source", ["raw", "data_uri", "path"])
def test_every_image_source_becomes_a_data_uri(source, tmp_path):
    path = tmp_path / "pixel.png"
    path.write_bytes(PNG_BYTES)
    image = {
        "raw": RAW_BASE64,
        "data_uri": DATA_URI,
        "path": str(path),
    }[source]

    (messages,) = _requests(lambda: _agent().run(TASK, img=image))

    (sent,) = _image_urls(messages)
    assert sent.startswith("data:image/")
    assert sent.endswith(RAW_BASE64)


def test_arun_sends_the_image():
    agent = _agent()

    (messages,) = _requests(
        lambda: asyncio.run(agent.arun(TASK, img=URL))
    )

    assert _image_urls(messages) == [URL]


def test_every_loop_keeps_the_image():
    agent = _agent(max_loops=2)

    requests = _requests(lambda: agent.run(TASK, img=URL))

    assert len(requests) == 2
    assert all(
        _image_urls(messages) == [URL] for messages in requests
    )


def test_image_joins_the_newest_turn_after_prior_messages():
    prior = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]

    (messages,) = _requests(
        lambda: _agent().run(TASK, img=URL, messages=prior)
    )

    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
    ]
    assert isinstance(messages[0]["content"], str)
    assert _image_urls(messages[-1:]) == [URL]


def test_image_is_not_written_into_the_conversation():
    agent = _agent()

    _requests(lambda: agent.run(TASK, img=DATA_URI))

    history = str(agent.short_memory.conversation_history)
    assert RAW_BASE64 not in history


def test_model_without_vision_raises_instead_of_dropping_the_image():
    agent = _agent(model_name="gpt-3.5-turbo")

    with pytest.raises(
        AgentLLMError, match="does not support vision"
    ):
        _requests(lambda: agent.run(TASK, img=URL))
