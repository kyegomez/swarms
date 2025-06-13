import base64
from swarms.utils.data_to_text import data_to_text


def test_data_to_text_binary(tmp_path):
    binary_path = tmp_path / "image.png"
    binary_bytes = b"\x89PNG\r\n\x1a\n"
    binary_path.write_bytes(binary_bytes)
    encoded = data_to_text(str(binary_path))
    assert base64.b64decode(encoded) == binary_bytes


def test_data_to_text_text(tmp_path):
    text_path = tmp_path / "file.txt"
    text_content = "hello"
    text_path.write_text(text_content)
    assert data_to_text(str(text_path)) == text_content
