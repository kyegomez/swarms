import pytest
import json

from swarms.utils.str_to_dict import str_to_dict


class TestStrToDict:
    """Test cases for the str_to_dict function."""

    def test_valid_json_string(self):
        """Test converting a valid JSON string to dictionary."""
        result = str_to_dict('{"key": "value"}')
        assert result == {"key": "value"}

    def test_nested_json_string(self):
        """Test converting a nested JSON string."""
        result = str_to_dict('{"a": 1, "b": {"c": 2}}')
        assert result == {"a": 1, "b": {"c": 2}}

    def test_list_in_json_string(self):
        """Test converting JSON string containing a list."""
        result = str_to_dict('{"items": [1, 2, 3]}')
        assert result == {"items": [1, 2, 3]}

    def test_empty_json_object(self):
        """Test converting an empty JSON object."""
        result = str_to_dict("{}")
        assert result == {}

    def test_json_with_numbers(self):
        """Test converting JSON string with various number types."""
        result = str_to_dict('{"int": 42, "float": 3.14, "negative": -5}')
        assert result == {"int": 42, "float": 3.14, "negative": -5}

    def test_json_with_booleans(self):
        """Test converting JSON string with boolean values."""
        result = str_to_dict('{"true_val": true, "false_val": false}')
        assert result == {"true_val": True, "false_val": False}

    def test_json_with_null(self):
        """Test converting JSON string with null value."""
        result = str_to_dict('{"value": null}')
        assert result == {"value": None}

    def test_invalid_json_raises_error(self):
        """Test that invalid JSON raises JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            str_to_dict('{"invalid": json}')  # Invalid JSON

    def test_complex_nested_structure(self):
        """Test converting a complex nested JSON structure."""
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
        assert result["user"]["name"] == "John"
        assert result["user"]["age"] == 30
        assert result["tags"] == ["python", "testing"]
        assert result["metadata"] is None

    def test_retries_parameter(self):
        """Test that retries parameter works correctly."""
        # This should succeed on first try
        result = str_to_dict('{"test": 1}', retries=1)
        assert result == {"test": 1}

    def test_json_with_unicode_characters(self):
        """Test converting JSON string with unicode characters."""
        result = str_to_dict('{"emoji": "🐍", "text": "你好"}')
        assert result["emoji"] == "🐍"
        assert result["text"] == "你好"

