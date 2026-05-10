import pytest
from swarms.structs.concat import concat_strings


def test_concat_strings_basic():
    """Test basic string concatenation"""
    result = concat_strings(["hello", " ", "world"])
    assert result == "hello world"


def test_concat_strings_empty_list():
    """Test concatenation with empty list"""
    result = concat_strings([])
    assert result == ""


def test_concat_strings_single_element():
    """Test concatenation with single element"""
    result = concat_strings(["hello"])
    assert result == "hello"


def test_concat_strings_multiple_elements():
    """Test concatenation with multiple elements"""
    result = concat_strings(["a", "b", "c", "d", "e"])
    assert result == "abcde"


def test_concat_strings_with_special_characters():
    """Test concatenation with special characters"""
    result = concat_strings(["hello", "\n", "world", "\t", "!"])
    assert result == "hello\nworld\t!"


def test_concat_strings_not_list_raises_typeerror():
    """Test that non-list input raises TypeError"""
    with pytest.raises(TypeError, match="Input must be a list of strings"):
        concat_strings("not a list")


def test_concat_strings_non_string_element_raises_typeerror():
    """Test that list with non-string elements raises TypeError"""
    with pytest.raises(TypeError, match="All elements in the list must be strings"):
        concat_strings(["hello", 123, "world"])


def test_concat_strings_mixed_types_raises_typeerror():
    """Test that list with mixed types raises TypeError"""
    with pytest.raises(TypeError, match="All elements in the list must be strings"):
        concat_strings(["hello", None, "world"])


def test_concat_strings_with_numbers_raises_typeerror():
    """Test that list containing numbers raises TypeError"""
    with pytest.raises(TypeError, match="All elements in the list must be strings"):
        concat_strings([1, 2, 3])


def test_concat_strings_empty_strings():
    """Test concatenation with empty strings"""
    result = concat_strings(["", "", ""])
    assert result == ""


def test_concat_strings_unicode():
    """Test concatenation with unicode characters"""
    result = concat_strings(["Hello", " ", "世界", " ", "🌍"])
    assert result == "Hello 世界 🌍"
