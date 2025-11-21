import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from loguru import logger
from swarms.structs.custom_agent import CustomAgent, AgentResponse

try:
    import pytest_asyncio

    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False
    pytest_asyncio = None


def create_test_custom_agent():
    return CustomAgent(
        name="TestAgent",
        description="Test agent for unit testing",
        base_url="https://api.test.com",
        endpoint="v1/test",
        headers={"Authorization": "Bearer test-token"},
        timeout=10.0,
        verify_ssl=True,
    )


@pytest.fixture
def sample_custom_agent():
    return create_test_custom_agent()


def test_custom_agent_initialization():
    try:
        custom_agent_instance = CustomAgent(
            name="TestAgent",
            description="Test description",
            base_url="https://api.example.com",
            endpoint="v1/endpoint",
            headers={"Content-Type": "application/json"},
            timeout=30.0,
            verify_ssl=True,
        )
        assert (
            custom_agent_instance.base_url
            == "https://api.example.com"
        )
        assert custom_agent_instance.endpoint == "v1/endpoint"
        assert custom_agent_instance.timeout == 30.0
        assert custom_agent_instance.verify_ssl is True
        assert "Content-Type" in custom_agent_instance.default_headers
        logger.info("CustomAgent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize CustomAgent: {e}")
        raise


def test_custom_agent_initialization_with_default_headers(
    sample_custom_agent,
):
    try:
        custom_agent_no_headers = CustomAgent(
            name="TestAgent",
            description="Test",
            base_url="https://api.test.com",
            endpoint="test",
        )
        assert (
            "Content-Type" in custom_agent_no_headers.default_headers
        )
        assert (
            custom_agent_no_headers.default_headers["Content-Type"]
            == "application/json"
        )
        logger.debug("Default Content-Type header added correctly")
    except Exception as e:
        logger.error(f"Failed to test default headers: {e}")
        raise


def test_custom_agent_url_normalization():
    try:
        custom_agent_with_slashes = CustomAgent(
            name="TestAgent",
            description="Test",
            base_url="https://api.test.com/",
            endpoint="/v1/test",
        )
        assert (
            custom_agent_with_slashes.base_url
            == "https://api.test.com"
        )
        assert custom_agent_with_slashes.endpoint == "v1/test"
        logger.debug("URL normalization works correctly")
    except Exception as e:
        logger.error(f"Failed to test URL normalization: {e}")
        raise


def test_prepare_headers(sample_custom_agent):
    try:
        prepared_headers = sample_custom_agent._prepare_headers()
        assert "Authorization" in prepared_headers
        assert (
            prepared_headers["Authorization"] == "Bearer test-token"
        )

        additional_headers = {"X-Custom-Header": "custom-value"}
        prepared_headers_with_additional = (
            sample_custom_agent._prepare_headers(additional_headers)
        )
        assert (
            prepared_headers_with_additional["X-Custom-Header"]
            == "custom-value"
        )
        assert (
            prepared_headers_with_additional["Authorization"]
            == "Bearer test-token"
        )
        logger.debug("Header preparation works correctly")
    except Exception as e:
        logger.error(f"Failed to test prepare_headers: {e}")
        raise


def test_prepare_payload_dict(sample_custom_agent):
    try:
        payload_dict = {"key": "value", "number": 123}
        prepared_payload = sample_custom_agent._prepare_payload(
            payload_dict
        )
        assert isinstance(prepared_payload, str)
        parsed = json.loads(prepared_payload)
        assert parsed["key"] == "value"
        assert parsed["number"] == 123
        logger.debug("Dictionary payload prepared correctly")
    except Exception as e:
        logger.error(f"Failed to test prepare_payload with dict: {e}")
        raise


def test_prepare_payload_string(sample_custom_agent):
    try:
        payload_string = '{"test": "value"}'
        prepared_payload = sample_custom_agent._prepare_payload(
            payload_string
        )
        assert prepared_payload == payload_string
        logger.debug("String payload prepared correctly")
    except Exception as e:
        logger.error(
            f"Failed to test prepare_payload with string: {e}"
        )
        raise


def test_prepare_payload_bytes(sample_custom_agent):
    try:
        payload_bytes = b'{"test": "value"}'
        prepared_payload = sample_custom_agent._prepare_payload(
            payload_bytes
        )
        assert prepared_payload == payload_bytes
        logger.debug("Bytes payload prepared correctly")
    except Exception as e:
        logger.error(
            f"Failed to test prepare_payload with bytes: {e}"
        )
        raise


def test_parse_response_success(sample_custom_agent):
    try:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"message": "success"}'
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"message": "success"}

        parsed_response = sample_custom_agent._parse_response(
            mock_response
        )
        assert isinstance(parsed_response, AgentResponse)
        assert parsed_response.status_code == 200
        assert parsed_response.success is True
        assert parsed_response.json_data == {"message": "success"}
        assert parsed_response.error_message is None
        logger.debug("Successful response parsed correctly")
    except Exception as e:
        logger.error(f"Failed to test parse_response success: {e}")
        raise


def test_parse_response_error(sample_custom_agent):
    try:
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_response.headers = {"content-type": "text/plain"}

        parsed_response = sample_custom_agent._parse_response(
            mock_response
        )
        assert isinstance(parsed_response, AgentResponse)
        assert parsed_response.status_code == 404
        assert parsed_response.success is False
        assert parsed_response.error_message == "HTTP 404"
        logger.debug("Error response parsed correctly")
    except Exception as e:
        logger.error(f"Failed to test parse_response error: {e}")
        raise


def test_extract_content_openai_format(sample_custom_agent):
    try:
        openai_response = {
            "choices": [
                {
                    "message": {
                        "content": "This is the response content"
                    }
                }
            ]
        }
        extracted_content = sample_custom_agent._extract_content(
            openai_response
        )
        assert extracted_content == "This is the response content"
        logger.debug("OpenAI format content extracted correctly")
    except Exception as e:
        logger.error(
            f"Failed to test extract_content OpenAI format: {e}"
        )
        raise


def test_extract_content_anthropic_format(sample_custom_agent):
    try:
        anthropic_response = {
            "content": [
                {"text": "First part "},
                {"text": "second part"},
            ]
        }
        extracted_content = sample_custom_agent._extract_content(
            anthropic_response
        )
        assert extracted_content == "First part second part"
        logger.debug("Anthropic format content extracted correctly")
    except Exception as e:
        logger.error(
            f"Failed to test extract_content Anthropic format: {e}"
        )
        raise


def test_extract_content_generic_format(sample_custom_agent):
    try:
        generic_response = {"text": "Generic response text"}
        extracted_content = sample_custom_agent._extract_content(
            generic_response
        )
        assert extracted_content == "Generic response text"
        logger.debug("Generic format content extracted correctly")
    except Exception as e:
        logger.error(
            f"Failed to test extract_content generic format: {e}"
        )
        raise


@patch("swarms.structs.custom_agent.httpx.Client")
def test_run_success(mock_client_class, sample_custom_agent):
    try:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = (
            '{"choices": [{"message": {"content": "Success"}}]}'
        )
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Success"}}]
        }
        mock_response.headers = {"content-type": "application/json"}

        mock_client_instance = Mock()
        mock_client_instance.__enter__ = Mock(
            return_value=mock_client_instance
        )
        mock_client_instance.__exit__ = Mock(return_value=None)
        mock_client_instance.post.return_value = mock_response
        mock_client_class.return_value = mock_client_instance

        test_payload = {"message": "test"}
        result = sample_custom_agent.run(test_payload)

        assert result == "Success"
        logger.info("Run method executed successfully")
    except Exception as e:
        logger.error(f"Failed to test run success: {e}")
        raise


@patch("swarms.structs.custom_agent.httpx.Client")
def test_run_error_response(mock_client_class, sample_custom_agent):
    try:
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client_instance = Mock()
        mock_client_instance.__enter__ = Mock(
            return_value=mock_client_instance
        )
        mock_client_instance.__exit__ = Mock(return_value=None)
        mock_client_instance.post.return_value = mock_response
        mock_client_class.return_value = mock_client_instance

        test_payload = {"message": "test"}
        result = sample_custom_agent.run(test_payload)

        assert "Error: HTTP 500" in result
        logger.debug("Error response handled correctly")
    except Exception as e:
        logger.error(f"Failed to test run error response: {e}")
        raise


@patch("swarms.structs.custom_agent.httpx.Client")
def test_run_request_error(mock_client_class, sample_custom_agent):
    try:
        import httpx

        mock_client_instance = Mock()
        mock_client_instance.__enter__ = Mock(
            return_value=mock_client_instance
        )
        mock_client_instance.__exit__ = Mock(return_value=None)
        mock_client_instance.post.side_effect = httpx.RequestError(
            "Connection failed"
        )
        mock_client_class.return_value = mock_client_instance

        test_payload = {"message": "test"}
        result = sample_custom_agent.run(test_payload)

        assert "Request error" in result
        logger.debug("Request error handled correctly")
    except Exception as e:
        logger.error(f"Failed to test run request error: {e}")
        raise


@pytest.mark.skipif(
    not ASYNC_AVAILABLE, reason="pytest-asyncio not installed"
)
@pytest.mark.asyncio
@patch("swarms.structs.custom_agent.httpx.AsyncClient")
async def test_run_async_success(
    mock_async_client_class, sample_custom_agent
):
    try:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = (
            '{"content": [{"text": "Async Success"}]}'
        )
        mock_response.json.return_value = {
            "content": [{"text": "Async Success"}]
        }
        mock_response.headers = {"content-type": "application/json"}

        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__ = AsyncMock(
            return_value=mock_client_instance
        )
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client_instance.post = AsyncMock(
            return_value=mock_response
        )
        mock_async_client_class.return_value = mock_client_instance

        test_payload = {"message": "test"}
        result = await sample_custom_agent.run_async(test_payload)

        assert result == "Async Success"
        logger.info("Run_async method executed successfully")
    except Exception as e:
        logger.error(f"Failed to test run_async success: {e}")
        raise


@pytest.mark.skipif(
    not ASYNC_AVAILABLE, reason="pytest-asyncio not installed"
)
@pytest.mark.asyncio
@patch("swarms.structs.custom_agent.httpx.AsyncClient")
async def test_run_async_error_response(
    mock_async_client_class, sample_custom_agent
):
    try:
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__ = AsyncMock(
            return_value=mock_client_instance
        )
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client_instance.post = AsyncMock(
            return_value=mock_response
        )
        mock_async_client_class.return_value = mock_client_instance

        test_payload = {"message": "test"}
        result = await sample_custom_agent.run_async(test_payload)

        assert "Error: HTTP 400" in result
        logger.debug("Async error response handled correctly")
    except Exception as e:
        logger.error(f"Failed to test run_async error response: {e}")
        raise


def test_agent_response_dataclass():
    try:
        agent_response_instance = AgentResponse(
            status_code=200,
            content="Success",
            headers={"content-type": "application/json"},
            json_data={"key": "value"},
            success=True,
            error_message=None,
        )
        assert agent_response_instance.status_code == 200
        assert agent_response_instance.content == "Success"
        assert agent_response_instance.success is True
        assert agent_response_instance.error_message is None
        logger.debug("AgentResponse dataclass created correctly")
    except Exception as e:
        logger.error(f"Failed to test AgentResponse dataclass: {e}")
        raise
