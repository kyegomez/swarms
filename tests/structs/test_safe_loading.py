"""SafeStateManager must be able to load back what it saved.

Agent.load() raised AttributeError for *every* agent: create_state_dict
serialises instance state including class-level read-only properties, and
load_state then tried to setattr them back. Agent has two — `workspace` and
`mcp_enabled` — so the very first one hit ended the load.

Nothing caught it because the only save/load round-trip test in the repo,
TestBasicAgent::test_save_and_load, has errored at setup since 2025-10-21
on a fixture that was deleted out from under it (#2010).

Offline: the LLM is a stub, and the workspace is a tmp_path.
"""

import os

import pytest

from swarms import Agent
from swarms.structs.safe_loading import SafeLoaderUtils


class _StubLLM:
    def run(self, task=None, *args, **kwargs):
        return task


def _agent(tmp_path, name, **overrides):
    kwargs = dict(
        agent_name=name,
        llm=_StubLLM(),
        max_loops=1,
        print_on=False,
        verbose=False,
        persistent_memory=False,
        autosave=False,
        workspace_dir=str(tmp_path),
    )
    kwargs.update(overrides)
    return Agent(**kwargs)


def test_agent_has_read_only_properties_in_its_state():
    """The premise: if this ever stops holding, the guard is unneeded."""
    read_only = [
        name
        for name, value in vars(Agent).items()
        if isinstance(value, property) and value.fset is None
    ]

    assert read_only, "Agent no longer has read-only properties"
    for name in read_only:
        assert not SafeLoaderUtils.is_settable(Agent, name)


def test_is_settable_allows_ordinary_attributes():
    assert SafeLoaderUtils.is_settable(Agent, "max_loops")
    assert SafeLoaderUtils.is_settable(Agent, "agent_name")


def test_save_then_load_does_not_raise(tmp_path):
    """The reported crash: load() blew up on the first read-only property."""
    path = str(tmp_path / "state.json")
    _agent(tmp_path, "saver").save(path)

    assert os.path.exists(path)

    # Fails on unfixed code with:
    #   AttributeError: property 'workspace' of 'Agent' object has no setter
    _agent(tmp_path, "loader").load(path)


def test_scalar_state_actually_round_trips(tmp_path):
    """Not just "does not raise" — the saved values come back."""
    path = str(tmp_path / "state.json")
    _agent(tmp_path, "saver", max_loops=3).save(path)

    restored = _agent(tmp_path, "loader", max_loops=9)
    restored.load(path)

    assert restored.max_loops == 3
    assert restored.agent_name == "saver"


def test_read_only_properties_survive_the_load(tmp_path):
    """Skipping them must not blank them out."""
    path = str(tmp_path / "state.json")
    _agent(tmp_path, "saver").save(path)

    restored = _agent(tmp_path, "loader")
    before = restored.workspace
    restored.load(path)

    assert restored.workspace == before


def test_missing_file_still_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _agent(tmp_path, "loader").load(str(tmp_path / "nope.json"))
