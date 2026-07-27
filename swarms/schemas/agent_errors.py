"""
Exception hierarchy for :class:`swarms.structs.agent.Agent`.

These live here rather than in ``agent.py`` so that agent collaborators
(``LLMManager``, ``SkillsManager``, ``MCPManager``, …) can raise and catch them
without importing ``Agent`` itself, which would be circular. They remain
importable from ``swarms.structs.agent`` for backwards compatibility.
"""


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


class AgentToolError(AgentError):
    """Exception raised when the agent fails to utilize a tool. Check the tool's configuration and availability."""

    pass


class AgentMemoryError(AgentError):
    """Exception raised when the agent encounters a memory-related issue. Ensure that memory resources are properly allocated and accessible."""

    pass


class AgentLLMInitializationError(AgentError):
    """Exception raised when the LLM fails to initialize properly. Please check the configuration and parameters."""

    pass


class AgentToolExecutionError(AgentError):
    """Exception raised when the agent fails to execute a tool. Check the tool's configuration and availability."""

    pass
