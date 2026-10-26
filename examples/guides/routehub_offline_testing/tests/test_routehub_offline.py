import asyncio
import json
import socket

import pytest
import routehub
from openai.types.chat import ChatCompletionChunk


MODEL = "gpt-5.4-mini"
MESSAGES = [
    {"role": "user", "content": "Return the review decision."}
]


def chunk_text(chunk: ChatCompletionChunk) -> str:
    """Extract text while allowing role, finish, and usage chunks."""
    if not chunk.choices:
        return ""
    return chunk.choices[0].delta.content or ""


@pytest.mark.parametrize(
    "reply", ["Approved.", "", "Approved.  Next step.\n"]
)
def test_completion_content(reply: str) -> None:
    """Preserve the text and stop reason, including empty replies."""
    result = routehub.completion(
        model=MODEL, messages=MESSAGES, mock_response=reply
    )
    assert result.choices[0].message.content == reply
    assert result.choices[0].finish_reason == "stop"


@pytest.mark.parametrize("reply", ["", "Approved.  Next step.\n"])
def test_stream_with_usage_chunk(reply: str) -> None:
    """Reassemble a stream without indexing an empty choices list."""
    with routehub.completion(
        model=MODEL,
        messages=MESSAGES,
        mock_response=reply,
        stream=True,
        stream_options={"include_usage": True},
    ) as stream:
        chunks = list(stream)
    assert "".join(chunk_text(chunk) for chunk in chunks) == reply
    assert chunks[-1].choices == []
    assert chunks[-1].usage is not None
    assert any(
        chunk.choices and chunk.choices[0].finish_reason == "stop"
        for chunk in chunks
    )


def test_async_completion() -> None:
    """Await a non-streaming mock response."""

    async def run() -> None:
        result = await routehub.acompletion(
            model=MODEL, messages=MESSAGES, mock_response="Approved."
        )
        assert result.choices[0].message.content == "Approved."

    asyncio.run(run())


@pytest.mark.parametrize("reply", ["", "Approved after review."])
def test_async_stream(reply: str) -> None:
    """Consume asynchronous streams, including empty replies."""

    async def run() -> None:
        stream = await routehub.acompletion(
            model=MODEL,
            messages=MESSAGES,
            mock_response=reply,
            stream=True,
        )
        async with stream:
            parts = [chunk_text(chunk) async for chunk in stream]
        assert "".join(parts) == reply

    asyncio.run(run())


def test_application_catches_rate_limit() -> None:
    """Catch the exception supplied to mock_response unchanged."""
    simulated = routehub.RateLimitError("simulated limit")
    with pytest.raises(routehub.RateLimitError) as caught:
        routehub.completion(
            model=MODEL, messages=MESSAGES, mock_response=simulated
        )
    assert caught.value is simulated
    assert caught.value.status_code == 429


@pytest.mark.parametrize("reply", ['{"approved": true}', "not JSON"])
def test_application_parses_json(reply: str) -> None:
    """Check application parsing of valid and malformed JSON."""
    result = routehub.completion(
        model=MODEL, messages=MESSAGES, mock_response=reply
    )
    content = result.choices[0].message.content
    assert isinstance(content, str)
    if reply == "not JSON":
        with pytest.raises(json.JSONDecodeError):
            json.loads(content)
    else:
        assert json.loads(content) == {"approved": True}


def test_network_guard_is_active() -> None:
    """Verify the guard raises before connecting."""
    with pytest.raises(
        AssertionError, match="Unexpected network access"
    ):
        socket.create_connection(("example.invalid", 443))
