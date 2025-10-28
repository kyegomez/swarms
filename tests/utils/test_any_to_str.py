import pytest

from loguru import logger
from swarms.utils.any_to_str import any_to_str


class TestAnyToStr:
    """Test cases for the any_to_str function."""

    def test_dictionary(self):
        """Test converting a dictionary to string."""
        try:
            result = any_to_str({"a": 1, "b": 2})
            assert result is not None, "Result should not be None"
            assert "a: 1" in result
            assert "b: 2" in result
        except Exception as e:
            logger.error(f"Error in test_dictionary: {e}")
            pytest.fail(f"test_dictionary failed with error: {e}")

    def test_list(self):
        """Test converting a list to string."""
        try:
            result = any_to_str([1, 2, 3])
            assert result is not None, "Result should not be None"
            assert "1" in result
            assert "2" in result
            assert "3" in result
            assert "[" in result or "," in result
        except Exception as e:
            logger.error(f"Error in test_list: {e}")
            pytest.fail(f"test_list failed with error: {e}")

    def test_none_value(self):
        """Test converting None to string."""
        try:
            result = any_to_str(None)
            assert result is not None, "Result should not be None"
            assert result == "None"
        except Exception as e:
            logger.error(f"Error in test_none_value: {e}")
            pytest.fail(f"test_none_value failed with error: {e}")

    def test_nested_dictionary(self):
        """Test converting a nested dictionary."""
        try:
            data = {
                "user": {
                    "id": 123,
                    "details": {"city": "New York", "active": True},
                },
                "data": [1, 2, 3],
            }
            result = any_to_str(data)
            assert result is not None, "Result should not be None"
            assert "user:" in result
            assert "data:" in result
        except Exception as e:
            logger.error(f"Error in test_nested_dictionary: {e}")
            pytest.fail(f"test_nested_dictionary failed with error: {e}")

    def test_tuple(self):
        """Test converting a tuple to string."""
        try:
            result = any_to_str((True, False, None))
            assert result is not None, "Result should not be None"
            assert "True" in result or "true" in result.lower()
            assert "(" in result or "," in result
        except Exception as e:
            logger.error(f"Error in test_tuple: {e}")
            pytest.fail(f"test_tuple failed with error: {e}")

    def test_empty_list(self):
        """Test converting an empty list."""
        try:
            result = any_to_str([])
            assert result is not None, "Result should not be None"
            assert result == "[]"
        except Exception as e:
            logger.error(f"Error in test_empty_list: {e}")
            pytest.fail(f"test_empty_list failed with error: {e}")

    def test_empty_dict(self):
        """Test converting an empty dictionary."""
        try:
            result = any_to_str({})
            assert result is not None, "Result should not be None"
            # Empty dict might return empty string or other representation
            assert isinstance(result, str)
        except Exception as e:
            logger.error(f"Error in test_empty_dict: {e}")
            pytest.fail(f"test_empty_dict failed with error: {e}")

    def test_string_with_quotes(self):
        """Test converting a string - should add quotes."""
        try:
            result = any_to_str("hello")
            assert result is not None, "Result should not be None"
            assert result == '"hello"'
        except Exception as e:
            logger.error(f"Error in test_string_with_quotes: {e}")
            pytest.fail(f"test_string_with_quotes failed with error: {e}")

    def test_integer(self):
        """Test converting an integer."""
        try:
            result = any_to_str(42)
            assert result is not None, "Result should not be None"
            assert result == "42"
        except Exception as e:
            logger.error(f"Error in test_integer: {e}")
            pytest.fail(f"test_integer failed with error: {e}")

    def test_mixed_types_in_list(self):
        """Test converting a list with mixed types."""
        try:
            result = any_to_str([1, "text", None, 2.5])
            assert result is not None, "Result should not be None"
            assert "1" in result
            assert "text" in result or '"text"' in result
            assert "None" in result
        except Exception as e:
            logger.error(f"Error in test_mixed_types_in_list: {e}")
            pytest.fail(f"test_mixed_types_in_list failed with error: {e}")
