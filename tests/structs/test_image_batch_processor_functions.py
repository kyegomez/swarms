import pytest
from swarms.structs.image_batch_processor import (
    ImageProcessingError,
    InvalidAgentError,
)


def test_image_processing_error_is_exception():
    """Test that ImageProcessingError is an Exception subclass"""
    assert issubclass(ImageProcessingError, Exception)


def test_invalid_agent_error_is_exception():
    """Test that InvalidAgentError is an Exception subclass"""
    assert issubclass(InvalidAgentError, Exception)


def test_image_processing_error_can_be_raised():
    """Test that ImageProcessingError can be raised and caught"""
    with pytest.raises(ImageProcessingError, match="Image error"):
        raise ImageProcessingError("Image error")


def test_invalid_agent_error_can_be_raised():
    """Test that InvalidAgentError can be raised and caught"""
    with pytest.raises(InvalidAgentError, match="Agent error"):
        raise InvalidAgentError("Agent error")


def test_image_processing_error_with_custom_message():
    """Test ImageProcessingError with custom message"""
    error = ImageProcessingError("Failed to process image")
    assert str(error) == "Failed to process image"


def test_invalid_agent_error_with_custom_message():
    """Test InvalidAgentError with custom message"""
    error = InvalidAgentError("Invalid agent configuration")
    assert str(error) == "Invalid agent configuration"


def test_image_processing_error_inheritance():
    """Test ImageProcessingError inheritance"""
    error = ImageProcessingError("Test")
    assert isinstance(error, Exception)
    assert isinstance(error, ImageProcessingError)


def test_invalid_agent_error_inheritance():
    """Test InvalidAgentError inheritance"""
    error = InvalidAgentError("Test")
    assert isinstance(error, Exception)
    assert isinstance(error, InvalidAgentError)


def test_image_processing_error_can_be_caught_as_exception():
    """Test that ImageProcessingError can be caught as Exception"""
    try:
        raise ImageProcessingError("Test error")
    except Exception as e:
        assert isinstance(e, ImageProcessingError)
        assert str(e) == "Test error"


def test_invalid_agent_error_can_be_caught_as_exception():
    """Test that InvalidAgentError can be caught as Exception"""
    try:
        raise InvalidAgentError("Test error")
    except Exception as e:
        assert isinstance(e, InvalidAgentError)
        assert str(e) == "Test error"
