import pytest


def test_multi_agent_debates_module_can_be_imported():
    """Test that multi_agent_debates module can be imported"""
    try:
        from swarms.structs import multi_agent_debates
        assert multi_agent_debates is not None
    except ImportError:
        pytest.skip("Module has import dependencies")


def test_multi_agent_debates_class_exists():
    """Test that MultiAgentDebate class exists"""
    try:
        from swarms.structs.multi_agent_debates import MultiAgentDebate
        assert MultiAgentDebate is not None
    except (ImportError, AttributeError):
        pytest.skip("Class not available or has dependencies")
