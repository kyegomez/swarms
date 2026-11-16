import base64
import os

import pytest

from swarms.utils.image_file_b64 import get_image_base64


@pytest.mark.parametrize("size", [300, 301, 302])
def test_raw_base64_becomes_a_data_uri(size):
    raw = base64.b64encode(os.urandom(size)).decode()

    assert get_image_base64(raw) == f"data:image/jpeg;base64,{raw}"
