import inspect
import pytest
from swarms.structs.agent import Agent
from swarms.structs.concurrent_workflow import ConcurrentWorkflow


def test_auto_save_removed_from_signature():
    """auto_save must be removed from ConcurrentWorkflow.__init__ parameter signature."""
    sig = inspect.signature(ConcurrentWorkflow.__init__)
    assert "auto_save" not in sig.parameters, "auto_save should not be in __init__ signature"
    assert "autosave" in sig.parameters, "autosave must be in __init__ signature"
    assert sig.parameters["autosave"].default is False


def test_auto_save_attribute_not_set_by_default():
    """ConcurrentWorkflow instance must not have auto_save attribute; only autosave."""
    agent = Agent(agent_name="test-agent", max_loops=1)
    wf = ConcurrentWorkflow(name="test-wf", agents=[agent])
    assert not hasattr(wf, "auto_save"), "auto_save attribute must be removed"
    assert hasattr(wf, "autosave")
    assert wf.autosave is False


def test_auto_save_deprecated_kwarg_warns_and_sets_autosave():
    """Passing auto_save should emit DeprecationWarning and populate autosave."""
    agent = Agent(agent_name="test-agent", max_loops=1)
    with pytest.deprecated_call():
        wf = ConcurrentWorkflow(
            name="test-wf-deprecated",
            agents=[agent],
            auto_save=True,
        )
    assert wf.autosave is True
