import base64
import os

import pytest

from swarms.utils import image_file_b64, litellm_wrapper
from swarms.utils.image_file_b64 import get_image_base64

PDF_URI = "data:application/pdf;base64,JVBERi0="


@pytest.mark.parametrize("size", [300, 301, 302])
def test_raw_base64_becomes_a_data_uri(size):
    raw = base64.b64encode(os.urandom(size)).decode()

    assert get_image_base64(raw) == f"data:image/jpeg;base64,{raw}"


@pytest.mark.parametrize(
    "source",
    [
        "data:APPLICATION/PDF;name=a.pdf;base64,JVBERi0=",
        "https://a.example/b.pdf?sig=1",
    ],
)
def test_pdf_is_sent_as_a_file_part(monkeypatch, source):
    monkeypatch.setattr(
        image_file_b64, "_fetch_image_url", lambda url: b"%PDF-"
    )
    monkeypatch.setattr(
        litellm_wrapper, "_model_supports_vision", lambda m: True
    )
    llm = litellm_wrapper.LiteLLM(model_name="claude-sonnet-4-6")

    assert llm._build_image_block(source) == {
        "type": "file",
        "file": {"file_data": PDF_URI, "filename": "document.pdf"},
    }


@pytest.mark.parametrize(
    "source", ["data:text/csv;base64,aGk=", "data:application/pdf,x"]
)
def test_unsupported_data_uri_is_rejected(source):
    with pytest.raises(ValueError, match="Unsupported data URI"):
        get_image_base64(source)
