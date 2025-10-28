import pytest
import json

from loguru import logger
from swarms.utils.str_to_dict import str_to_dict


class TestStrToDict:
    """Test cases for the str_to_dict function."""

    def test_valid_json_string(self):
        """Test converting a valid JSON string to dictionary."""
        try:
            result = str_to_dict('{"key": "value"}')
            assert result is not None, "Result should not be None"
            assert result == {"key": "value"}
        except Exception as e:
            logger.error(f"Error in test_valid_json_string: {e}")
            pytest.fail(f"test_valid_json_string failed with error: {e}")

    def test_nested_json_string(self):
        """Test converting a nested JSON string."""
        try:
            result = str_to_dict('{"a": 1, "b": {"c": 2}}')
            assert result is not None, "Result should not be None"
            assert result == {"a": 1, "b": {"c": 2}}
        except Exception as e:
            logger.error(f"Error in test_nested_json_string: {e}")
            pytest.fail(f"test_nested_json_string failed with error: {e}")

    def test_list_in_json_string(self):
        """Test converting JSON string containing a list."""
        try:
            result = str_to_dict('{"items": [1, 2, 3]}')
            assert result is not None, "Result should not be None"
            assert result == {"items": [1, 2, 3]}
        except Exception as e:
            logger.error(f"Error in test_list_in_json_string: {e}")
            pytest.fail(f"test_list_in_json_string failed with error: {e}")

    def test_empty_json_object(self):
        """Test converting an empty JSON object."""
        try:
            result = str_to_dict("{}")
            assert result is not None, "Result should not be None"
            assert result == {}
        except Exception as e:
            logger.error(f"Error in test_empty_json_object: {e}")
            pytest.fail(f"test_empty_json_object failed with error: {e}")

    def test_json_with_numbers(self):
        """Test converting JSON string with various number types."""
        try:
            result = str_to_dict('{"int": 42, "float": 3.14, "negative": -5}')
            assert result is not None, "Result should not be None"
            assert result == {"int": 42, "float": 3.14, "negative": -5}
        except Exception as e:
            logger.error(f"Error in test_json_with_numbers: {e}")
            pytest.fail(f"test_json_with_numbers failed with error: {e}")

    def test_json_with_booleans(self):
        """Test converting JSON string with boolean values."""
        try:
            result = str_to_dict('{"true_val": true, "false_val": false}')
            assert result is not None, "Result should not be None"
            assert result == {"true_val": True, "false_val": False}
        except Exception as e:
            logger.error(f"Error in test_json_with_booleans: {e}")
            pytest.fail(f"test_json_with_booleans failed with error: {e}")

    def test_json_with_null(self):
        """Test converting JSON string with null value."""
        try:
            result = str_to_dict('{"value": null}')
            assert result is not None, "Result should not be None"
            assert result == {"value": None}
        except Exception as e:
            logger.error(f"Error in test_json_with_null: {e}")
            pytest.fail(f"test_json_with_null failed with error: {e}")

    def test_invalid_json_raises_error(self):
        """Test that invalid JSON raises JSONDecodeError."""
        try:
            with pytest.raises(json.JSONDecodeError):
                str_to_dict('{"invalid": json}')  # Invalid JSON
        except Exception as e:
            logger.error(f"Error in test_invalid_json_raises_error: {e}")
            pytest.fail(f"test_invalid_json_raises_error failed with error: {e}")

    def test_complex_nested_structure(self):
        """Test converting a complex nested JSON structure."""
        try:
            json_str = '''
            {
                "user": {
                    "name": "John",
                    "age": 30,
                    "active": true
                },
                "tags": ["python", "testing"],
                "metadata": null
            }
            '''
            result = str_to_dict(json_str)
            assert result is not None, "Result should not be None"
            assert result["user"]["name"] == "John"
            assert result["user"]["age"] == 30
            assert result["tags"] == ["python", "testing"]
            assert result["metadata"] is None
        except Exception as e:
            logger.error(f"Error in test_complex_nested_structure: {e}")
            pytest.fail(f"test_complex_nested_structure failed with error: {e}")

    def test_retries_parameter(self):
        """Test that retries parameter works correctly."""
        try:
            # This should succeed on first try
            result = str_to_dict('{"test": 1}', retries=1)
            assert result is not None, "Result should not be None"
            assert result == {"test": 1}
        except Exception as e:
            logger.error(f"Error in test_retries_parameter: {e}")
            pytest.fail(f"test_retries_parameter failed with error: {e}")

    def test_json_with_unicode_characters(self):
        """Test converting JSON string with unicode characters."""
        try:
            result = str_to_dict('{"emoji": "🐍", "text": "你好"}')
            assert result is not None, "Result should not be None"
            assert result["emoji"] == "🐍"
            assert result["text"] == "你好"
        except Exception as e:
            logger.error(f"Error in test_json_with_unicode_characters: {e}")
            pytest.fail(f"test_json_with_unicode_characters failed with error: {e}")
