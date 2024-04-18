# Filename: test_utils.py

import os

import pytest

from swarms.utils import find_image_path


def test_find_image_path_no_images():
    assert (
        find_image_path("This is a test string without any image paths.")
        is None
    )


def test_find_image_path_one_image():
    text = "This is a string with one image path: sample_image.jpg."
    assert find_image_path(text) == "sample_image.jpg"


def test_find_image_path_multiple_images():
    text = "This string has two image paths: img1.png, and img2.jpg."
    assert (
        find_image_path(text) == "img2.jpg"
    )  # Assuming both images exist


def test_find_image_path_wrong_input():
    with pytest.raises(TypeError):
        find_image_path(123)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("no image path here", None),
        ("image: sample.png", "sample.png"),
        ("image: sample.png, another: another.jpeg", "another.jpeg"),
    ],
)
def test_find_image_path_parameterized(text, expected):
    assert find_image_path(text) == expected


def mock_os_path_exists(path):
    return True


def test_find_image_path_mocking(monkeypatch):
    monkeypatch.setattr(os.path, "exists", mock_os_path_exists)
    assert find_image_path("image.jpg") == "image.jpg"
