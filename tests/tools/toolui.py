from unittest import result
import gradio_client
import pytest
import os
from unittest.mock import patch, MagicMock
from app import set_environ, load_tools, download_model

def test_set_environ():

    @patch('app.LLM')
    def test_download_model(mock_llm):
    # Arrange
        model_url = "facebook/opt-125m"
        memory_utilization = 8
        mock_model = MagicMock()
        mock_llm.return_value = mock_model

        # Act
        result = download_model(model_url, memory_utilization)

        # Assert
        mock_llm.assert_called_once_with(model=model_url, trust_remote_code=True, gpu_memory_utilization=memory_utilization)
        self.assertEqual(result, gradio_client.update(choices=[(model_url.split('/')[-1], mock_model)]))

    def test_load_tools(self):
        # Call the function
        result = load_tools()

# Check if the function returns the expected result
assert result is not None
assert isinstance(result, list)
