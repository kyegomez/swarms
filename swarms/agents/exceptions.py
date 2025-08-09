from typing import Any, Dict, Optional


class ToolAgentError(Exception):
    """Base exception for all tool agent errors."""

    def __init__(
        self, message: str, details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ToolExecutionError(ToolAgentError):
    """Raised when a tool fails to execute."""

    def __init__(
        self,
        tool_name: str,
        error: Exception,
        details: Optional[Dict[str, Any]] = None,
    ):
        message = (
            f"Failed to execute tool '{tool_name}': {str(error)}"
        )
        super().__init__(message, details)


class ToolValidationError(ToolAgentError):
    """Raised when tool parameters fail validation."""

    def __init__(
        self,
        tool_name: str,
        param_name: str,
        error: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        message = f"Validation error for tool '{tool_name}' parameter '{param_name}': {error}"
        super().__init__(message, details)


class ToolNotFoundError(ToolAgentError):
    """Raised when a requested tool is not found."""

    def __init__(
        self, tool_name: str, details: Optional[Dict[str, Any]] = None
    ):
        message = f"Tool '{tool_name}' not found"
        super().__init__(message, details)


class ToolParameterError(ToolAgentError):
    """Raised when tool parameters are invalid."""

    def __init__(
        self,
        tool_name: str,
        error: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        message = (
            f"Invalid parameters for tool '{tool_name}': {error}"
        )
        super().__init__(message, details)
