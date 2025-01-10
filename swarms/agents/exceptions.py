from enum import Enum
from typing import Dict, Optional

class ErrorSeverity(Enum):
    """Enum for error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ToolAgentError(Exception):
    """Base exception class for ToolAgent errors."""
    def __init__(
        self, 
        message: str, 
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict] = None
    ):
        self.severity = severity
        self.details = details or {}
        super().__init__(message)

class ValidationError(ToolAgentError):
    """Raised when validation fails."""
    pass

class ModelNotProvidedError(ToolAgentError):
    """Raised when neither model nor llm is provided."""
    pass

class SecurityError(ToolAgentError):
    """Raised when security checks fail."""
    pass

class SchemaValidationError(ToolAgentError):
    """Raised when JSON schema validation fails."""
    pass

class ConfigurationError(ToolAgentError):
    """Raised when there's an error in configuration."""
    pass