import pytest

from swarms.utils import extract_code_from_markdown


@pytest.fixture
def markdown_content_with_code():
    return """
    # This is a markdown document
    
    Some intro text here.
Some additional text.
"""


@pytest.fixture
def markdown_content_without_code():
    return """
    # This is a markdown document

    There is no code in this document.
    """


def test_extract_code_from_markdown_with_code(
    markdown_content_with_code,
):
    extracted_code = extract_code_from_markdown(markdown_content_with_code)
    assert "def my_func():" in extracted_code
    assert 'print("This is my function.")' in extracted_code
    assert "class MyClass:" in extracted_code
    assert "pass" in extracted_code


def test_extract_code_from_markdown_without_code(
    markdown_content_without_code,
):
    extracted_code = extract_code_from_markdown(
        markdown_content_without_code
    )
    assert extracted_code == ""


def test_extract_code_from_markdown_exception():
    with pytest.raises(TypeError):
        extract_code_from_markdown(None)
