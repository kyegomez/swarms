"""
Tests for ``swarms.utils.workspace_manager.WorkspaceManager``.

Every test points WORKSPACE_DIR at a tmp_path, so nothing touches the real
workspace. No network and no LLM: the manager only ever touches the disk.
"""

import json
import os

import pytest

from swarms.utils.workspace_manager import (
    CONVERSATION_FILENAME,
    agent_dir_name,
    WorkspaceManager,
    conversation_to_data,
    ensure_workspace_env,
    sanitize_name,
)
from swarms.utils.workspace_utils import get_workspace_dir


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    """Point WORKSPACE_DIR at a tmp dir and clear the lru_cache."""
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))
    get_workspace_dir.cache_clear()
    yield tmp_path
    get_workspace_dir.cache_clear()


class FakeConversation:
    def __init__(self, history=None):
        self.conversation_history = (
            history
            if history is not None
            else [{"role": "user", "content": "hi"}]
        )


class DictOnlyConversation:
    def to_dict(self):
        return [{"role": "assistant", "content": "from to_dict"}]


class FakeSwarm:
    def __init__(self, name="test-swarm", conversation=None):
        self.name = name
        self.id = "swarm-id-1"
        self.max_loops = 2
        self.agents = [object(), object()]
        self.conversation = conversation or FakeConversation()


class TestSanitizeName:
    def test_empty_becomes_unnamed(self):
        assert sanitize_name("") == "unnamed"
        assert sanitize_name(None) == "unnamed"

    def test_path_separators_cannot_escape_the_directory(self):
        """A name with a slash must not create a nested path."""
        assert "/" not in sanitize_name("evil/../../etc")
        assert "\\" not in sanitize_name("evil\\win")

    def test_spaces_become_hyphens(self):
        assert sanitize_name("my swarm") == "my-swarm"

    def test_long_names_are_truncated(self):
        assert len(sanitize_name("x" * 500)) == 100

    def test_name_of_only_dots_does_not_become_empty(self):
        """'..' would otherwise strip to '' and yield a bare path."""
        assert sanitize_name("..") == "unnamed"


class TestEnsureWorkspaceEnv:
    def test_returns_the_configured_dir(self, workspace):
        assert ensure_workspace_env() == str(workspace)

    def test_defaults_when_unset(self, monkeypatch, tmp_path):
        monkeypatch.delenv("WORKSPACE_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        get_workspace_dir.cache_clear()
        resolved = ensure_workspace_env()
        assert resolved.endswith("agent_workspace")
        assert os.environ["WORKSPACE_DIR"] == resolved


class TestConversationToData:
    def test_none_is_empty(self):
        assert conversation_to_data(None) == []

    def test_reads_conversation_history(self):
        conv = FakeConversation([{"role": "user", "content": "a"}])
        assert conversation_to_data(conv) == [
            {"role": "user", "content": "a"}
        ]

    def test_falls_back_to_to_dict(self):
        data = conversation_to_data(DictOnlyConversation())
        assert data[0]["content"] == "from to_dict"

    def test_object_with_neither_is_empty(self):
        assert conversation_to_data(object()) == []


class TestSetup:
    def test_directory_layout(self, workspace):
        wm = WorkspaceManager(FakeSwarm(name="my-swarm"))
        rel = os.path.relpath(wm.dir, workspace)
        parts = rel.split(os.sep)
        assert parts[0] == "swarms"
        assert parts[1] == "FakeSwarm"
        assert parts[2].startswith("my-swarm-")
        assert os.path.isdir(wm.dir)

    def test_disabled_creates_nothing(self, workspace):
        wm = WorkspaceManager(FakeSwarm(), enabled=False)
        assert wm.dir is None
        assert not bool(wm)
        assert not (workspace / "swarms").exists()

    def test_name_argument_overrides_owner_name(self):
        wm = WorkspaceManager(
            FakeSwarm(name="owner"), name="override"
        )
        assert os.path.basename(wm.dir).startswith("override-")

    def test_uuid_mode(self):
        wm = WorkspaceManager(FakeSwarm(), use_timestamp=False)
        stamp = os.path.basename(wm.dir).rsplit("-", 1)[1]
        assert len(stamp) == 12

    def test_two_managers_do_not_collide(self):
        a = WorkspaceManager(FakeSwarm(), use_timestamp=False)
        b = WorkspaceManager(FakeSwarm(), use_timestamp=False)
        assert a.dir != b.dir


class TestSaves:
    def test_save_conversation_writes_history(self):
        swarm = FakeSwarm()
        wm = WorkspaceManager(swarm)
        path = wm.save_conversation()

        assert os.path.basename(path) == CONVERSATION_FILENAME
        with open(path) as f:
            assert json.load(f) == [{"role": "user", "content": "hi"}]

    def test_explicit_conversation_beats_the_owner(self):
        wm = WorkspaceManager(FakeSwarm())
        path = wm.save_conversation(
            FakeConversation([{"role": "x", "content": "explicit"}])
        )
        with open(path) as f:
            assert json.load(f)[0]["content"] == "explicit"

    def test_save_all_writes_four_files(self):
        wm = WorkspaceManager(FakeSwarm())
        paths = wm.save_all(execution_result=["a", "b"])

        assert set(paths) == {
            "config",
            "state",
            "metadata",
            "conversation",
        }
        assert all(paths.values())
        assert sorted(os.listdir(wm.dir)) == [
            "config.json",
            CONVERSATION_FILENAME,
            "metadata.json",
            "state.json",
        ]

    def test_metadata_summarizes_rather_than_embeds(self):
        wm = WorkspaceManager(FakeSwarm())
        with open(wm.save_metadata(execution_result=["a", "b"])) as f:
            meta = json.load(f)
        assert meta["execution_result_summary"] == {
            "type": "list",
            "length": 2,
        }
        assert meta["agents_count"] == 2

    def test_unserializable_values_do_not_raise(self):
        """default=str must absorb objects json cannot encode."""
        swarm = FakeSwarm()
        swarm.conversation = FakeConversation([{"obj": object()}])
        wm = WorkspaceManager(swarm)
        assert wm.save_conversation() is not None

    def test_disabled_manager_writes_nothing(self):
        wm = WorkspaceManager(FakeSwarm(), enabled=False)
        assert wm.save_conversation() is None
        assert wm.save_config() is None
        assert wm.save_state() is None
        assert wm.save_metadata() is None


class TestNeverRaises:
    def test_setup_failure_disables_rather_than_raises(
        self, monkeypatch
    ):
        def boom(*a, **kw):
            raise OSError("read-only filesystem")

        monkeypatch.setattr(os, "makedirs", boom)
        wm = WorkspaceManager(FakeSwarm())
        assert wm.dir is None
        assert wm.save_conversation() is None

    def test_write_failure_returns_none(self, monkeypatch):
        wm = WorkspaceManager(FakeSwarm())

        def boom(*a, **kw):
            raise OSError("disk full")

        monkeypatch.setattr("builtins.open", boom)
        assert wm.save_conversation() is None

    def test_owner_without_optional_attributes(self):
        """A bare owner must still produce config/state/metadata."""

        class Bare:
            pass

        wm = WorkspaceManager(Bare(), name="bare")
        assert wm.save_config() is not None
        assert wm.save_state() is not None
        assert wm.save_metadata() is not None


class TestAgentLayout:
    """Agents keep the older ``agents/{name}-{uuid}`` path, not ``swarms/``."""

    @staticmethod
    def _legacy_dir_name(agent_name, agent_id):
        """The exact algorithm agent.py used before the migration."""
        if agent_name:
            s = (
                agent_name.lower()
                .replace(" ", "-")
                .replace("_", "-")
                .replace("/", "-")
                .replace("\\", "-")
                .replace(":", "-")
                .replace("*", "-")
                .replace("?", "-")
                .replace('"', "-")
                .replace("<", "-")
                .replace(">", "-")
                .replace("|", "-")
                .replace("--", "-")
                .replace("--", "-")
                .strip("-")
            )
        else:
            s = "agent"
        u = (
            agent_id.replace("agent-", "")
            if agent_id.startswith("agent-")
            else agent_id
        )
        return f"{s}-{u[-12:] if len(u) > 12 else u}"

    @pytest.mark.parametrize(
        "name,agent_id",
        [
            ("My Agent", "agent-abcdef123456789"),
            ("weird/name:with*chars", "agent-0011223344556677"),
            ("Multi__Under___scores", "xyz"),
            ("", "agent-deadbeefcafe"),
            ("Trailing---", "agent-1234"),
            ("a", "short"),
        ],
    )
    def test_matches_the_pre_migration_naming(self, name, agent_id):
        """The path is load-bearing: file tools resolve against it."""
        assert agent_dir_name(
            name, agent_id
        ) == self._legacy_dir_name(name, agent_id)

    def test_for_agent_uses_the_agents_directory(self, workspace):
        class FakeAgent:
            agent_name = "My Agent"
            id = "agent-abcdef123456789"

        wm = WorkspaceManager.for_agent(FakeAgent())
        rel = os.path.relpath(wm.dir, workspace).split(os.sep)
        assert rel[0] == "agents"
        assert rel[1] == "my-agent-def123456789"
        assert "swarms" not in rel

    def test_agent_metadata_keys(self):
        class FakeAgent:
            agent_name = "Saver"
            id = "agent-1234567890abcdef"

        wm = WorkspaceManager.for_agent(FakeAgent())
        with open(
            wm.save_config(additional_metadata={"loop_count": 3})
        ) as f:
            meta = json.load(f)["_autosave_metadata"]

        assert meta["agent_name"] == "Saver"
        assert meta["agent_id"] == "agent-1234567890abcdef"
        assert meta["loop_count"] == 3
        assert "swarm_name" not in meta


class TestAtomicWrite:
    def test_no_temp_file_is_left_behind(self):
        wm = WorkspaceManager(FakeSwarm())
        wm.save_conversation()
        assert not [
            f for f in os.listdir(wm.dir) if f.endswith(".tmp")
        ]

    def test_a_failed_write_leaves_the_previous_file_intact(
        self, monkeypatch
    ):
        """The point of temp+replace: config.json is rewritten each loop."""
        wm = WorkspaceManager(FakeSwarm())
        path = wm.save_conversation()
        original = open(path).read()

        real_replace = os.replace

        def boom(*a, **kw):
            raise OSError("interrupted")

        monkeypatch.setattr(os, "replace", boom)
        assert wm.save_conversation() is None
        monkeypatch.setattr(os, "replace", real_replace)

        assert open(path).read() == original
