import sys
from unittest import result
import gradio as gr
import gradio_client
import pytest
import os
from unittest.mock import patch, MagicMock
from app import answer_by_tools, deploy_on_sky_pilot, fetch_tokenizer, model_chosen
from swarms.app import start_tool_server
from swarms.app import set_environ, load_tools, download_model
import pytest
from gradio_client import *
from swarms.modelui.modules.models import load_model
from swarms.tools.tool_controller import ToolInfo
import dotenv

os.environ.get("")

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, ".."))

HOST = "localhost"
PORT = 8000

# TODO fix this test

def test_set_environ():
    @patch("app.LLM")
    def test_download_model(mock_llm):
        # Arrange
        model_url = "facebook/opt-125m"
        memory_utilization = 8
        mock_model = MagicMock()
        mock_llm.return_value = mock_model

        # Act
        result = download_model(model_url, memory_utilization)

        # Assert
        mock_llm.assert_called_once_with(
            model=model_url,
            trust_remote_code=True,
            gpu_memory_utilization=memory_utilization,
        )
        assert (
            result,
            gradio_client.update(choices=[(model_url.split("/")[-1], mock_model)]),
        )

def test_load_tools(self):
    # Call the function
    result = load_tools()
    # Assert
    assert result is not None
    assert isinstance(result, list)
    
@pytest.mark.unit
def test_start_tool_server():
    # Mock server creation
    with patch("socket.socket") as mock_socket:
        start_tool_server()

        # Assert socket creation and port binding
        mock_socket.assert_called_once()
        mock_socket().bind.assert_called_once_with((HOST, PORT))

@pytest.mark.unit
def test_tools():
    # Define a list of tools to test
    tools_to_test = ["file_operation_tool", "tool2", "tool3"]

    for tool in tools_to_test:
        # Mock tool information access
        with patch("gradio_interface.VALID_TOOLS_INFO", {tool: {"desc": "description"}}):
            tools = load_tools()

            # Assert tool list creation
            assert tools == [ToolInfo(tool, "description")]
        
@pytest.mark.unit
def test_answer_by_tools():
    # Mock user input and tool/model availability
    with patch("gradio_interface.get_user_input", return_value="question"):
        with patch("gradio_interface.are_tools_available", return_value=True):
            # Mock selected model and tools
            selected_model = MagicMock()
            selected_tools = [MagicMock(), MagicMock()]
            # Call the function with the selected model and tools
            answer = answer_by_tools(selected_model, selected_tools)

            # Assert that the function can get a response
            assert answer is not None

@pytest.mark.unit
@patch("app.answer_by_tools")
def test_answer_by_tools_mocked(answer_by_tools_mock):
    # Mock user input and tool/model availability
    with patch("gradio_interface.get_user_input", return_value="question"):
        with patch("gradio_interface.are_tools_available", return_value=True):
            # Mock selected model and tools
            selected_model = MagicMock()
            selected_tools = [MagicMock(), MagicMock()]

            # Call the function with the selected model and tools
            answer = answer_by_tools(selected_model, selected_tools)

            # Assert that the function can get a response
            assert answer is not None

            # Assert tool/model calls and return values
            answer_by_tools_mock.assert_called_once_with(selected_model, selected_tools)

@pytest.mark.integration
def test_deployment_and_usage_of_vllm_models():
    # Mock SkyPilot and tool responses
    with patch("sky.Task") as mock_sky_task:
        with patch("sky.launch") as mock_sky_launch:
            deploy_on_sky_pilot("model_name", "tokenizer", "A100")

            # Assert SkyPilot deployment
            mock_sky_task.assert_called_once()
            mock_sky_launch.assert_called_once()

    # Mock AWS and tokenizer fetch
    with patch("boto3.client") as mock_boto3_client:
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:
            
            set_environ(AWS_ACCESS_KEY_ID="aws_access_key_id", AWS_SECRET_ACCESS_KEY="aws_secret_access_key", AWS_DEFAULT_REGION="aws_default_region")
            fetch_tokenizer("model_name")

            # Assert AWS and tokenizer fetch
            mock_boto3_client.assert_called_once()
            mock_tokenizer.assert_called_once()

@pytest.mark.e2e
def test_gradio_interface():
    # Mock Gradio interface and user interaction
    with patch("gradio.Interface") as mock_gradio_interface:
        with patch("gradio.Textbox") as mock_gradio_textbox:
            answer_by_tools("question", ["tool1"], "model1")

            # Assert Gradio interface and user interaction
            mock_gradio_interface.assert_called_once()
            mock_gradio_textbox.assert_called_once()

    # Get available models and tools from Gradio API
    available_models = gradio_client.get_available_models()
    available_tools = gradio_client.get_available_tools()

    # Select a model and tool
    selected_model = model_chosen(available_models)
    selected_tool = model_chosen(available_tools)

    # Use the selected model and tool
    selected_model.use()
    selected_tool.use()

    # TODO Alt model methods
    # Use the loaded model and tool
    # model.use()
    # tool.use()

# Check if the function returns the expected result
assert result is not None
assert isinstance(result, list)
