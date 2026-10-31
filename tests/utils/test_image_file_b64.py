import base64
import os
from types import SimpleNamespace

import pytest

from swarms.utils import image_file_b64, litellm_wrapper, llm_backend
from swarms.utils.image_file_b64 import get_image_base64
from swarms.utils.litellm_wrapper import LiteLLM

PDF_URI = "data:application/pdf;base64,JVBERi0="
PNG_URI = "data:image/png;base64,iVBORw0KGgo="
PNG_PART = {
    "type": "image_url",
    "image_url": {"url": PNG_URI, "format": "image/png"},
}
PDF_PART = {
    "type": "file",
    "file": {"file_data": PDF_URI, "filename": "document.pdf"},
}


@pytest.mark.parametrize("size", [300, 301, 302])
def test_raw_base64_becomes_a_data_uri(size):
    raw = base64.b64encode(os.urandom(size)).decode()

    assert get_image_base64(raw) == f"data:image/jpeg;base64,{raw}"


@pytest.mark.parametrize(
    "model, source, part",
    [
        ("gpt-4o", PNG_URI, PNG_PART),
        ("gpt-4o", PDF_URI, PDF_PART),
        ("gpt-4o", "minutes.pdf", PDF_PART),
        ("gpt-4o", "https://a.example/b.pdf?sig=1", PDF_PART),
        ("anthropic/claude-sonnet-4-20250514", PDF_URI, PDF_PART),
    ],
)
def test_pdf_input_is_sent_as_a_file_part(
    tmp_path, monkeypatch, model, source, part
):
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content="ok", tool_calls=None)
        choice = SimpleNamespace(
            message=message, finish_reason="stop"
        )
        return SimpleNamespace(choices=[choice], usage=None)

    monkeypatch.setattr(litellm_wrapper, "completion", completion)
    monkeypatch.setattr(
        llm_backend,
        "supports_vision",
        lambda model: True,
        raising=False,
    )
    monkeypatch.setattr(
        image_file_b64, "_fetch_image_url", lambda url: b"%PDF-"
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "minutes.pdf").write_bytes(b"%PDF-")

    LiteLLM(model_name=model).run("Read this", img=source)

    assert calls[-1]["messages"][-1]["content"] == [
        {"type": "text", "text": "Read this"},
        part,
    ]
