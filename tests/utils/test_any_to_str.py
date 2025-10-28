import pytest

from swarms.utils.any_to_str import any_to_str


class TestAnyToStr:
    """Test cases for the any_to_str function."""

    def test_dictionary(self):
        """Test converting a dictionary to string."""
        result = any_to_str({"a": 1, "b": 2})
        assert "a: 1" in result
        assert "b: 2" in result

    def test_list(self):
        """Test converting a list to string."""
        result = any_to_str([1, 2, 3])
        assert "1" in result
        assert "2" in result
        assert "3" in result
        assert "[" in result or "," in result

    def test_none_value(self):
        """Test converting None to string."""
        result = any_to_str(None)
        assert result == "None"

    def test_nested_dictionary(self):
        """Test converting a nested dictionary."""
        data = {
            "user": {
                "id": 123,
                "details": {"city": "New York", "active": True},
            },
            "data": [1, 2, 3],
        }
        result = any_to_str(data)
        assert "user:" in result
        assert "data:" in result

    def test_tuple(self):
        """Test converting a tuple to string."""
        result = any_to_str((True, False, None))
        assert "True" in result or "true" in result.lower()
        assert "(" in result or "," in result

    def test_empty_list(self):
        """Test converting an empty list."""
        result = any_to_str([])
        assert result == "[]"

    def test_empty_dict(self):
        """Test converting an empty dictionary."""
        result = any_to_str({})
        assert result == "" or "None" in result or len(result.strip()) == 0

    def test_string_with_quotes(self):
        """Test converting a string - should add quotes."""
        result = any_to_str("hello")
        assert result == '"hello"'

    def test_integer(self):
        """Test converting an integer."""
        result = any_to_str(42)
        assert result == "42"

    def test_mixed_types_in_list(self):
        """Test converting a list with mixed types."""
        result = any_to_str([1, "text", None, 2.5])
        assert "1" in result
        assert "text" in result or '"text"' in result
        assert "None" in result

