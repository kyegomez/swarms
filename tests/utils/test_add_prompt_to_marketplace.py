"""
Pytest tests for swarms_marketplace_utils module.
"""
import os
from unittest.mock import Mock, patch

import pytest

from swarms.utils.swarms_marketplace_utils import (
    add_prompt_to_marketplace,
)


class TestAddPromptToMarketplace:
    """Test cases for add_prompt_to_marketplace function."""

    @patch.dict(os.environ, {"SWARMS_API_KEY": "test_api_key_12345"})
    @patch("swarms.utils.swarms_marketplace_utils.httpx.Client")
    def test_add_prompt_success(self, mock_client_class):
        """Test successful addition of prompt to marketplace."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "123",
            "name": "Blood Analysis Agent",
            "status": "success",
        }
        mock_response.text = ""
        mock_response.raise_for_status = Mock()

        # Mock client
        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Call function
        result = add_prompt_to_marketplace(
            name="Blood Analysis Agent",
            prompt="You are a blood analysis agent that can analyze blood samples and provide a report on the results.",
            description="A blood analysis agent that can analyze blood samples and provide a report on the results.",
            use_cases=[
                {
                    "title": "Blood Analysis",
                    "description": "Analyze blood samples and provide a report on the results.",
                }
            ],
            tags="blood, analysis, report",
            category="research",
        )

        # Assertions
        assert result["id"] == "123"
        assert result["name"] == "Blood Analysis Agent"
        assert result["status"] == "success"
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://swarms.world/api/add-prompt"
        assert call_args[1]["headers"]["Authorization"] == "Bearer test_api_key_12345"
        assert call_args[1]["json"]["name"] == "Blood Analysis Agent"
        assert call_args[1]["json"]["category"] == "research"

    @patch.dict(os.environ, {"SWARMS_API_KEY": "test_api_key_12345"})
    @patch("swarms.utils.swarms_marketplace_utils.httpx.Client")
    def test_add_prompt_with_all_parameters(self, mock_client_class):
        """Test adding prompt with all optional parameters."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "456", "status": "success"}
        mock_response.text = ""
        mock_response.raise_for_status = Mock()

        # Mock client
        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Call function with all parameters
        result = add_prompt_to_marketplace(
            name="Test Prompt",
            prompt="Test prompt text",
            description="Test description",
            use_cases=[{"title": "Use Case 1", "description": "Description 1"}],
            tags="tag1, tag2",
            is_free=False,
            price_usd=9.99,
            category="coding",
            timeout=60.0,
        )

        # Assertions
        assert result["id"] == "456"
        call_args = mock_client.post.call_args
        json_data = call_args[1]["json"]
        assert json_data["is_free"] is False
        assert json_data["price_usd"] == 9.99
        assert json_data["category"] == "coding"
        assert json_data["tags"] == "tag1, tag2"

    def test_add_prompt_missing_api_key(self):
        """Test that missing API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Swarms API key is not set"):
                add_prompt_to_marketplace(
                    name="Test",
                    prompt="Test prompt",
                    description="Test description",
                    use_cases=[],
                    category="research",
                )

    def test_add_prompt_empty_api_key(self):
        """Test that empty API key raises ValueError."""
        with patch.dict(os.environ, {"SWARMS_API_KEY": ""}):
            with pytest.raises(ValueError, match="Swarms API key is not set"):
                add_prompt_to_marketplace(
                    name="Test",
                    prompt="Test prompt",
                    description="Test description",
                    use_cases=[],
                    category="research",
                )

    def test_add_prompt_missing_name(self):
        """Test that missing name raises ValueError."""
        with patch.dict(os.environ, {"SWARMS_API_KEY": "test_key"}):
            with pytest.raises(ValueError, match="name is required"):
                add_prompt_to_marketplace(
                    name=None,
                    prompt="Test prompt",
                    description="Test description",
                    use_cases=[],
                    category="research",
                )

    def test_add_prompt_missing_prompt(self):
        """Test that missing prompt raises ValueError."""
        with patch.dict(os.environ, {"SWARMS_API_KEY": "test_key"}):
            with pytest.raises(ValueError, match="prompt is required"):
                add_prompt_to_marketplace(
                    name="Test",
                    prompt=None,
                    description="Test description",
                    use_cases=[],
                    category="research",
                )

    def test_add_prompt_missing_description(self):
        """Test that missing description raises ValueError."""
        with patch.dict(os.environ, {"SWARMS_API_KEY": "test_key"}):
            with pytest.raises(ValueError, match="description is required"):
                add_prompt_to_marketplace(
                    name="Test",
                    prompt="Test prompt",
                    description=None,
                    use_cases=[],
                    category="research",
                )

    def test_add_prompt_missing_category(self):
        """Test that missing category raises ValueError."""
        with patch.dict(os.environ, {"SWARMS_API_KEY": "test_key"}):
            with pytest.raises(ValueError, match="category is required"):
                add_prompt_to_marketplace(
                    name="Test",
                    prompt="Test prompt",
                    description="Test description",
                    use_cases=[],
                    category=None,
                )

    def test_add_prompt_missing_use_cases(self):
        """Test that missing use_cases raises ValueError."""
        with patch.dict(os.environ, {"SWARMS_API_KEY": "test_key"}):
            with pytest.raises(ValueError, match="use_cases is required"):
                add_prompt_to_marketplace(
                    name="Test",
                    prompt="Test prompt",
                    description="Test description",
                    use_cases=None,
                    category="research",
                )

    @patch.dict(os.environ, {"SWARMS_API_KEY": "test_api_key_12345"})
    @patch("swarms.utils.swarms_marketplace_utils.httpx.Client")
    def test_add_prompt_http_error(self, mock_client_class):
        """Test handling of HTTP error responses."""
        # Mock response with error
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.reason_phrase = "Bad Request"
        mock_response.json.return_value = {"error": "Invalid request"}
        mock_response.text = '{"error": "Invalid request"}'
        mock_response.raise_for_status.side_effect = Exception("HTTP 400")

        # Mock client
        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Call function and expect exception
        with pytest.raises(Exception):
            add_prompt_to_marketplace(
                name="Test",
                prompt="Test prompt",
                description="Test description",
                use_cases=[],
                category="research",
            )

    @patch.dict(os.environ, {"SWARMS_API_KEY": "test_api_key_12345"})
    @patch("swarms.utils.swarms_marketplace_utils.httpx.Client")
    def test_add_prompt_authentication_error(self, mock_client_class):
        """Test handling of authentication errors."""
        # Mock response with 401 error
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.reason_phrase = "Unauthorized"
        mock_response.json.return_value = {
            "error": "Authentication failed"
        }
        mock_response.text = '{"error": "Authentication failed"}'
        mock_response.raise_for_status.side_effect = Exception("HTTP 401")

        # Mock client
        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Call function and expect exception
        with pytest.raises(Exception):
            add_prompt_to_marketplace(
                name="Test",
                prompt="Test prompt",
                description="Test description",
                use_cases=[],
                category="research",
            )

    @patch.dict(os.environ, {"SWARMS_API_KEY": "test_api_key_12345"})
    @patch("swarms.utils.swarms_marketplace_utils.httpx.Client")
    def test_add_prompt_with_empty_tags(self, mock_client_class):
        """Test adding prompt with empty tags."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "789", "status": "success"}
        mock_response.text = ""
        mock_response.raise_for_status = Mock()

        # Mock client
        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Call function with empty tags
        result = add_prompt_to_marketplace(
            name="Test",
            prompt="Test prompt",
            description="Test description",
            use_cases=[],
            tags=None,
            category="research",
        )

        # Assertions
        assert result["id"] == "789"
        call_args = mock_client.post.call_args
        assert call_args[1]["json"]["tags"] == ""

    @patch.dict(os.environ, {"SWARMS_API_KEY": "test_api_key_12345"})
    @patch("swarms.utils.swarms_marketplace_utils.httpx.Client")
    def test_add_prompt_request_timeout(self, mock_client_class):
        """Test handling of request timeout."""
        # Mock client to raise timeout error
        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client.post.side_effect = Exception("Request timeout")
        mock_client_class.return_value = mock_client

        # Call function and expect exception
        with pytest.raises(Exception):
            add_prompt_to_marketplace(
                name="Test",
                prompt="Test prompt",
                description="Test description",
                use_cases=[],
                category="research",
                timeout=5.0,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
