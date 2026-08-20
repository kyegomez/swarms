"""Tests for the Agent error-class re-exports.

``swarms.structs.agent`` re-exports every class from
``swarms.schemas.agent_errors`` so consumers can import errors from either
location and get the same object.
"""

import pytest

from swarms.schemas import agent_errors as schema_errors

ERROR_CLASS_NAMES = [
    "AgentError",
    "AgentInitializationError",
    "AgentRunError",
    "AgentLLMError",
    "AgentToolError",
    "AgentMemoryError",
    "AgentLLMInitializationError",
    "AgentToolExecutionError",
]


class TestErrorClassExports:
    """The schema error classes must be importable from swarms.structs.agent."""

    @pytest.mark.parametrize("class_name", ERROR_CLASS_NAMES)
    def test_importable_from_structs_agent(self, class_name):
        from_structs = __import__(
            "swarms.structs.agent", fromlist=[class_name]
        )
        assert getattr(from_structs, class_name) is getattr(
            schema_errors, class_name
        )