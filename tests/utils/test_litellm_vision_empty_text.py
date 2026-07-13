"""Regression tests for vision message construction in ``LiteLLM``.

An image-only vision call (empty, whitespace-only, or ``None`` task) must not
emit an empty text content block. Anthropic rejects such blocks with
"text content blocks must be non-empty"; the wrapper already guards this for
system blocks in ``_prepare_messages`` but historically emitted
``{"type": "text", "text": ""}`` for the vision user message.

These assertions are on the message structure the wrapper builds (they do not
make a live provider call).
"""

import base64

import pytest

from swarms.utils.litellm_wrapper import LiteLLM

# Smallest valid 1x1 PNG, enough for get_image_base64() to encode from a file
# without any network access.
_PNG_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4"
    "2mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@pytest.fixture
def png_path(tmp_path):
    path = tmp_path / "pixel.png"
    path.write_bytes(_PNG_1x1)
    return str(path)


def _user_content(messages):
    user = next(m for m in messages if m["role"] == "user")
    return user["content"]


def _text_blocks(content):
    return [
        b
        for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    ]


def _image_blocks(content):
    return [
        b
        for b in content
        if isinstance(b, dict) and b.get("type") == "image_url"
    ]


@pytest.mark.parametrize("task", ["", "   ", None])
def test_anthropic_vision_omits_empty_text_block(task, png_path):
    llm = LiteLLM(model_name="claude-3-5-sonnet-20241022")
    messages = llm.anthropic_vision_processing(task, png_path, [])
    content = _user_content(messages)
    assert _text_blocks(content) == []
    assert len(_image_blocks(content)) == 1


@pytest.mark.parametrize("task", ["", "   ", None])
def test_openai_vision_omits_empty_text_block(task, png_path):
    llm = LiteLLM(model_name="gpt-4o")
    messages = llm.openai_vision_processing(task, png_path, [])
    content = _user_content(messages)
    assert _text_blocks(content) == []
    assert len(_image_blocks(content)) == 1


def test_anthropic_direct_url_omits_empty_text_block(monkeypatch):
    # Force the direct-URL branch so the URL is embedded, not fetched.
    llm = LiteLLM(model_name="claude-3-5-sonnet-20241022")
    monkeypatch.setattr(
        llm, "_should_use_direct_url", lambda image: True
    )
    url = "https://example.com/pixel.png"
    messages = llm.anthropic_vision_processing("", url, [])
    content = _user_content(messages)
    assert _text_blocks(content) == []
    assert _image_blocks(content)[0]["image_url"]["url"] == url


def test_anthropic_vision_keeps_nonempty_text_block(png_path):
    # Over-deletion guard: a real task must still carry its text block.
    llm = LiteLLM(model_name="claude-3-5-sonnet-20241022")
    messages = llm.anthropic_vision_processing(
        "Describe this image", png_path, []
    )
    content = _user_content(messages)
    blocks = _text_blocks(content)
    assert len(blocks) == 1
    assert blocks[0]["text"] == "Describe this image"
    assert len(_image_blocks(content)) == 1


def test_openai_vision_keeps_nonempty_text_block(png_path):
    llm = LiteLLM(model_name="gpt-4o")
    messages = llm.openai_vision_processing(
        "Describe this image", png_path, []
    )
    content = _user_content(messages)
    blocks = _text_blocks(content)
    assert len(blocks) == 1
    assert blocks[0]["text"] == "Describe this image"
    assert len(_image_blocks(content)) == 1
