import base64
import os

import pytest

from swarms.utils.image_file_b64 import get_image_base64


@pytest.mark.parametrize("size", [300, 301, 302])
def test_raw_base64_becomes_a_data_uri(size):
    raw = base64.b64encode(os.urandom(size)).decode()

    assert get_image_base64(raw) == f"data:image/jpeg;base64,{raw}"


@pytest.mark.parametrize(
    "header, mime_type",
    [
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"GIF89a", "image/gif"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
        (b"\xff\xd8\xff\xe0", "image/jpeg"),
    ],
)
def test_raw_base64_is_labelled_with_its_real_format(
    header, mime_type
):
    raw = base64.b64encode(header + os.urandom(300)).decode()

    assert get_image_base64(raw) == f"data:{mime_type};base64,{raw}"
