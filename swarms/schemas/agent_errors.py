class AgentError(Exception):
    """Base class for all agent-related exceptions."""

    pass


class AgentInitializationError(AgentError):
    """Exception raised when the agent fails to initialize properly. Please check the configuration and parameters."""

    pass


class AgentRunError(AgentError):
    """Exception raised when the agent encounters an error during execution. Ensure that the task and environment are set up correctly."""

    pass


class AgentLLMError(AgentError):
    """Exception raised when there is an issue with the language model (LLM). Verify the model's availability and compatibility."""

    pass


class AgentLLMInitializationError(AgentError):
    """Exception raised when the LLM fails to initialize properly. Please check the configuration and parameters."""

    pass


class AgentToolExecutionError(AgentError):
    """Exception raised when the agent fails to execute a tool. Check the tool's configuration and availability."""

    pass
