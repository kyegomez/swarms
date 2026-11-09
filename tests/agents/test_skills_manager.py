"""
Test suite for :class:`swarms.agents.skills_manager.SkillsManager`.

Everything here runs against a real filesystem — no mocks:

* Skill directories are built under pytest's ``tmp_path`` fixture, each
  containing a real ``SKILL.md`` file with YAML frontmatter written to disk.
* ``os``, ``open`` and ``yaml`` are never mocked/patched — ``load_metadata``
  and ``load_full_skill`` are exercised through genuine file I/O.
* The dynamic-loading path is exercised through the real
  :class:`swarms.structs.dynamic_skills_loader.DynamicSkillsLoader`, which
  computes actual cosine similarity over real skill descriptions.
* The Agent-integration section constructs a real ``swarms.Agent`` to
  verify that ``skills_dir`` / ``skills_metadata`` / ``handle_skills`` wire
  through to the underlying ``SkillsManager`` correctly. The only stub is
  ``litellm.completion`` in the request tests, which capture what would be
  sent to the provider, so no network/API key is required.

Run:
    cd /Users/swarms_wd/Desktop/research/swarms
    PYTHONPATH=. python3 -m pytest tests/agents/test_skills_manager.py -v
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from swarms.agents.skills_manager import (
    SKILLS_PROMPT_HEADER,
    SkillsManager,
)
from swarms.structs.dynamic_skills_loader import DynamicSkillsLoader
from swarms.utils import litellm_wrapper


########################################################
# Helpers: build real skill folders on disk
########################################################


def _write_skill(
    root,
    folder_name: str,
    *,
    name=None,
    description="A helpful skill.",
    body="Do the thing carefully.",
    frontmatter_extra: str = "",
) -> None:
    """Create ``root/folder_name/SKILL.md`` with real YAML frontmatter."""
    skill_dir = root / folder_name
    skill_dir.mkdir(parents=True, exist_ok=True)

    lines = ["---"]
    if name is not None:
        lines.append(f"name: {name}")
    lines.append(f"description: {description}")
    if frontmatter_extra:
        lines.append(frontmatter_extra)
    lines.append("---")
    lines.append("")
    lines.append(body)

    (skill_dir / "SKILL.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


########################################################
# Fixtures
########################################################


@pytest.fixture
def skills_root(tmp_path):
    """A directory with two well-formed skills plus assorted noise."""
    root = tmp_path / "skills"
    root.mkdir()

    _write_skill(
        root,
        "pdf-processing",
        name="pdf-processing",
        description="Extract text and tables from PDF documents.",
        body="## Steps\n\n1. Open the PDF\n2. Extract text",
    )
    _write_skill(
        root,
        "web-search",
        name="web-search",
        description="Search the web for up to date information.",
        body="## Steps\n\n1. Formulate a query\n2. Search",
    )

    # A stray file, not a directory — must be skipped.
    (root / "NOTES.txt").write_text("not a skill", encoding="utf-8")

    # A directory with no SKILL.md — must be skipped.
    (root / "empty-folder").mkdir()

    return root


@pytest.fixture
def manager(skills_root):
    return SkillsManager(skills_dir=str(skills_root))


########################################################
# load_metadata()
########################################################


class TestLoadMetadata:
    def test_valid_skills_parsed(self, manager):
        skills = manager.load_metadata()
        by_name = {s["name"]: s for s in skills}

        assert set(by_name) == {"pdf-processing", "web-search"}

        pdf = by_name["pdf-processing"]
        assert (
            pdf["description"]
            == "Extract text and tables from PDF documents."
        )
        assert pdf["path"].endswith(
            os.path.join("pdf-processing", "SKILL.md")
        )
        assert "Open the PDF" in pdf["content"]

    def test_explicit_skills_dir_overrides_configured(self, tmp_path):
        configured_dir = tmp_path / "configured"
        configured_dir.mkdir()

        other_dir = tmp_path / "other"
        other_dir.mkdir()
        _write_skill(
            other_dir,
            "only-in-other",
            name="only-in-other",
            description="Lives in the other directory.",
        )

        mgr = SkillsManager(skills_dir=str(configured_dir))
        # Configured dir has nothing loadable.
        assert mgr.load_metadata() == []

        # Explicit argument overrides the configured directory.
        skills = mgr.load_metadata(skills_dir=str(other_dir))
        assert [s["name"] for s in skills] == ["only-in-other"]

        # And the configured dir was not mutated.
        assert mgr.skills_dir == str(configured_dir)

    def test_missing_directory_returns_empty_list(self, tmp_path):
        mgr = SkillsManager(
            skills_dir=str(tmp_path / "does-not-exist")
        )
        assert mgr.load_metadata() == []

    def test_no_skills_dir_configured_returns_empty_list(self):
        mgr = SkillsManager(skills_dir=None)
        assert mgr.load_metadata() == []

    def test_non_directory_entries_skipped(self, manager):
        names = {s["name"] for s in manager.load_metadata()}
        assert "NOTES.txt" not in names

    def test_folder_without_skill_md_skipped(self, manager):
        names = {s["name"] for s in manager.load_metadata()}
        assert "empty-folder" not in names

    def test_malformed_no_frontmatter_skipped(self, tmp_path):
        root = tmp_path / "skills"
        root.mkdir()
        skill_dir = root / "broken"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "Just a plain markdown file, no frontmatter at all.",
            encoding="utf-8",
        )

        mgr = SkillsManager(skills_dir=str(root))
        assert mgr.load_metadata() == []

    def test_malformed_unterminated_frontmatter_skipped(
        self, tmp_path
    ):
        root = tmp_path / "skills"
        root.mkdir()
        skill_dir = root / "broken"
        skill_dir.mkdir()
        # Starts with "---" but never closes it -> split("---", 2) yields
        # only 2 parts, which load_metadata treats as malformed.
        (skill_dir / "SKILL.md").write_text(
            "---\nname: broken\ndescription: no closing marker\n",
            encoding="utf-8",
        )

        mgr = SkillsManager(skills_dir=str(root))
        assert mgr.load_metadata() == []

    def test_malformed_invalid_yaml_skipped(self, tmp_path):
        root = tmp_path / "skills"
        root.mkdir()
        skill_dir = root / "broken"
        skill_dir.mkdir()
        # Unbalanced flow-mapping brackets -> yaml.safe_load raises.
        (skill_dir / "SKILL.md").write_text(
            "---\nname: [unterminated\n---\nbody text\n",
            encoding="utf-8",
        )

        mgr = SkillsManager(skills_dir=str(root))
        assert mgr.load_metadata() == []

    def test_malformed_unreadable_file_skipped(self, tmp_path):
        root = tmp_path / "skills"
        root.mkdir()
        skill_dir = root / "unreadable"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            "---\nname: unreadable\ndescription: x\n---\nbody",
            encoding="utf-8",
        )
        os.chmod(skill_file, 0o000)

        try:
            if os.access(skill_file, os.R_OK):
                pytest.skip(
                    "Running as a user that bypasses file permissions "
                    "(e.g. root) — cannot simulate an unreadable file."
                )
            mgr = SkillsManager(skills_dir=str(root))
            assert mgr.load_metadata() == []
        finally:
            os.chmod(skill_file, 0o644)

    def test_name_falls_back_to_folder_name(self, tmp_path):
        root = tmp_path / "skills"
        root.mkdir()
        _write_skill(
            root,
            "folder-as-name",
            name=None,
            description="No explicit name in frontmatter.",
        )

        mgr = SkillsManager(skills_dir=str(root))
        skills = mgr.load_metadata()
        assert len(skills) == 1
        assert skills[0]["name"] == "folder-as-name"

    def test_deterministic_ordering(self, tmp_path):
        root = tmp_path / "skills"
        root.mkdir()
        for folder in ["zzz-skill", "aaa-skill", "mmm-skill"]:
            _write_skill(
                root, folder, name=folder, description="desc"
            )

        mgr = SkillsManager(skills_dir=str(root))
        names = [s["name"] for s in mgr.load_metadata()]
        assert names == sorted(names)

        # Repeated calls are stable too.
        assert [s["name"] for s in mgr.load_metadata()] == names


########################################################
# build_prompt()
########################################################


class TestBuildPrompt:
    def test_empty_list_returns_empty_string(self, manager):
        assert manager.build_prompt([]) == ""

    def test_header_appears_exactly_once(self, manager):
        skills = manager.load_metadata()
        prompt = manager.build_prompt(skills)
        assert prompt.count(SKILLS_PROMPT_HEADER) == 1

    def test_every_skill_name_description_body_present(self, manager):
        skills = manager.load_metadata()
        prompt = manager.build_prompt(skills)

        for skill in skills:
            assert skill["name"] in prompt
            assert skill["description"] in prompt
            assert skill["content"] in prompt


########################################################
# prompt_for_task()
########################################################


class TestPromptForTask:
    def test_none_task_takes_static_path_loads_everything(
        self, manager
    ):
        prompt = manager.prompt_for_task(None)
        assert "pdf-processing" in prompt
        assert "web-search" in prompt
        assert len(manager.metadata) == 2

    def test_task_string_takes_dynamic_path(self, manager):
        prompt = manager.prompt_for_task(
            "Please extract text and tables from a PDF document."
        )
        assert "pdf-processing" in prompt
        assert manager.dynamic_loader is not None

    def test_no_skills_dir_configured_returns_empty_string(self):
        mgr = SkillsManager(skills_dir=None)
        assert mgr.prompt_for_task(None) == ""
        assert mgr.prompt_for_task("some task") == ""

    def test_dir_with_no_loadable_skills_returns_empty_string(
        self, tmp_path
    ):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        mgr = SkillsManager(skills_dir=str(empty_dir))

        assert mgr.prompt_for_task(None) == ""
        assert mgr.prompt_for_task("do something") == ""

    def test_metadata_populated_as_side_effect_static(self, manager):
        assert manager.metadata == []
        manager.prompt_for_task(None)
        assert len(manager.metadata) == 2

    def test_metadata_populated_as_side_effect_dynamic(self, manager):
        assert manager.metadata == []
        manager.prompt_for_task("search the web for news")
        assert any(
            s["name"] == "web-search" for s in manager.metadata
        )


########################################################
# Dynamic loading via DynamicSkillsLoader
########################################################


class TestDynamicLoading:
    def test_matching_task_selects_skill(self, manager):
        prompt = manager._dynamic_prompt(
            "Search the web for up to date information right now."
        )
        assert "web-search" in prompt
        assert "pdf-processing" not in prompt

    def test_unrelated_task_selects_nothing(self, manager):
        prompt = manager._dynamic_prompt(
            "Completely unrelated gibberish about zebras and volcanoes."
        )
        assert prompt == ""
        # metadata is only overwritten when something is found.
        assert manager.metadata == []

    def test_loader_built_lazily(self, manager):
        assert manager.dynamic_loader is None
        manager.prompt_for_task("extract text from a pdf document")
        assert isinstance(manager.dynamic_loader, DynamicSkillsLoader)

    def test_loader_reused_across_calls(self, manager):
        manager.prompt_for_task("extract text from a pdf document")
        first_loader = manager.dynamic_loader

        manager.prompt_for_task("search the web for news")
        assert manager.dynamic_loader is first_loader


########################################################
# set_skills_dir()
########################################################


class TestSetSkillsDir:
    def test_repoints_directory(self, manager, tmp_path):
        other_dir = tmp_path / "other"
        other_dir.mkdir()

        manager.set_skills_dir(str(other_dir))
        assert manager.skills_dir == str(other_dir)

    def test_discards_cached_metadata(self, manager):
        manager.prompt_for_task(None)
        assert manager.metadata != []

        manager.set_skills_dir(manager.skills_dir)
        assert manager.metadata == []

    def test_discards_cached_dynamic_loader(self, manager):
        manager.prompt_for_task("extract text from a pdf")
        assert manager.dynamic_loader is not None

        manager.set_skills_dir(manager.skills_dir)
        assert manager.dynamic_loader is None

    def test_set_to_none_disables(self, manager):
        manager.set_skills_dir(None)
        assert manager.skills_dir is None
        assert manager.enabled is False


########################################################
# enabled property
########################################################


class TestEnabled:
    def test_true_when_dir_set_and_exists(self, manager):
        assert manager.enabled is True

    def test_false_when_dir_not_set(self):
        mgr = SkillsManager(skills_dir=None)
        assert mgr.enabled is False

    def test_false_when_dir_set_but_missing(self, tmp_path):
        mgr = SkillsManager(skills_dir=str(tmp_path / "nonexistent"))
        assert mgr.enabled is False

    def test_false_for_empty_string_dir(self):
        mgr = SkillsManager(skills_dir="")
        assert mgr.enabled is False


########################################################
# load_full_skill()
########################################################


class TestLoadFullSkill:
    def test_returns_body_for_known_skill(self, manager):
        # load_full_skill reads from manager.metadata, which load_metadata()
        # does not populate as a side effect -- prompt_for_task does.
        manager.prompt_for_task(None)
        content = manager.load_full_skill("pdf-processing")
        assert content is not None
        assert "Open the PDF" in content

    def test_returns_none_for_unknown_skill(self, manager):
        manager.prompt_for_task(None)
        assert manager.load_full_skill("does-not-exist") is None

    def test_returns_none_when_file_deleted_after_metadata_load(
        self, manager, skills_root
    ):
        manager.prompt_for_task(None)
        skill_file = skills_root / "pdf-processing" / "SKILL.md"
        assert skill_file.exists()
        skill_file.unlink()

        # Must not raise, even though the path in metadata is now stale.
        assert manager.load_full_skill("pdf-processing") is None


########################################################
# Agent integration
########################################################


class TestAgentIntegration:
    def test_agent_skills_dir_wires_up_manager(self, skills_root):
        from swarms import Agent

        agent = Agent(
            agent_name="SkillsWiringAgent",
            model_name="gpt-4o-mini",
            skills_dir=str(skills_root),
            persistent_memory=False,
            print_on=False,
        )

        assert agent.skills.skills_dir == str(skills_root)
        assert agent.skills_dir == str(skills_root)

    def test_skills_dir_getter_reads_through(self, skills_root):
        from swarms import Agent

        agent = Agent(
            agent_name="SkillsGetterAgent",
            model_name="gpt-4o-mini",
            skills_dir=str(skills_root),
            persistent_memory=False,
            print_on=False,
        )

        # Mutating the underlying manager is visible through the property.
        agent.skills.set_skills_dir(None)
        assert agent.skills_dir is None

    def test_skills_dir_setter_writes_through(self, tmp_path):
        from swarms import Agent

        agent = Agent(
            agent_name="SkillsSetterAgent",
            model_name="gpt-4o-mini",
            persistent_memory=False,
            print_on=False,
        )

        new_dir = tmp_path / "new-skills"
        new_dir.mkdir()
        agent.skills_dir = str(new_dir)

        assert agent.skills.skills_dir == str(new_dir)

    def test_skills_metadata_getter_reads_through(self, skills_root):
        from swarms import Agent

        agent = Agent(
            agent_name="SkillsMetaGetterAgent",
            model_name="gpt-4o-mini",
            skills_dir=str(skills_root),
            persistent_memory=False,
            print_on=False,
        )

        agent.skills.metadata = agent.skills.load_metadata()
        assert agent.skills_metadata == agent.skills.metadata
        assert len(agent.skills_metadata) == 2

    def test_skills_metadata_setter_writes_through(self):
        from swarms import Agent

        agent = Agent(
            agent_name="SkillsMetaSetterAgent",
            model_name="gpt-4o-mini",
            persistent_memory=False,
            print_on=False,
        )

        fake_metadata = [{"name": "fake", "description": "d"}]
        agent.skills_metadata = fake_metadata
        assert agent.skills.metadata == fake_metadata

    def test_handle_skills_selects_all_skills_without_task(
        self, skills_root
    ):
        from swarms import Agent

        agent = Agent(
            agent_name="SkillsPromptAgent",
            model_name="gpt-4o-mini",
            skills_dir=str(skills_root),
            persistent_memory=False,
            print_on=False,
        )

        before = agent.system_prompt
        agent.handle_skills(task=None)

        assert agent.system_prompt == before
        assert "pdf-processing" in agent._skills_prompt
        assert "web-search" in agent._skills_prompt
        assert SKILLS_PROMPT_HEADER in agent._skills_prompt

    def test_handle_skills_with_task_selects_dynamic_skills(
        self, skills_root
    ):
        from swarms import Agent

        agent = Agent(
            agent_name="SkillsPromptDynamicAgent",
            model_name="gpt-4o-mini",
            skills_dir=str(skills_root),
            persistent_memory=False,
            print_on=False,
        )

        before = agent.system_prompt
        agent.handle_skills(
            task="Search the web for up to date information right now."
        )

        assert agent.system_prompt == before
        assert "web-search" in agent._skills_prompt

    def test_handle_skills_replaces_the_previous_selection(
        self, skills_root
    ):
        from swarms import Agent

        agent = Agent(
            agent_name="SkillsReplaceAgent",
            model_name="gpt-4o-mini",
            skills_dir=str(skills_root),
            persistent_memory=False,
            print_on=False,
        )

        agent.handle_skills(task=None)
        agent.handle_skills(
            task="Search the web for up to date information right now."
        )

        assert agent._skills_prompt.count(SKILLS_PROMPT_HEADER) == 1
        assert "pdf-processing" not in agent._skills_prompt


def _capture_requests():
    """Stub ``completion`` and collect the messages of every request."""
    requests = []
    message = SimpleNamespace(
        role="assistant", content="ok", tool_calls=None
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=message, finish_reason="stop")
        ],
        usage=None,
    )

    def fake_completion(**kwargs):
        requests.append(kwargs["messages"])
        return response

    stub = patch.object(
        litellm_wrapper, "completion", side_effect=fake_completion
    )
    return requests, stub


def _system_text(request):
    return "\n".join(
        str(m["content"]) for m in request if m["role"] == "system"
    )


def _user_text(request):
    return "\n".join(
        str(m["content"]) for m in request if m["role"] == "user"
    )


class TestSkillsReachTheModel:
    @pytest.fixture
    def agent(self, skills_root):
        from swarms import Agent

        return Agent(
            agent_name="SkillsRequestAgent",
            model_name="gpt-4o-mini",
            skills_dir=str(skills_root),
            max_loops=1,
            persistent_memory=False,
            print_on=False,
        )

    def test_run_sends_selected_skill_as_system_message(self, agent):
        requests, stub = _capture_requests()

        with stub:
            agent.run(
                "Search the web for up to date information right now."
            )

        system = _system_text(requests[-1])
        assert SKILLS_PROMPT_HEADER.strip() in system
        assert "web-search" in system

    def test_repeated_runs_send_the_skills_once(self, agent):
        requests, stub = _capture_requests()
        task = "Search the web for up to date information right now."

        with stub:
            agent.run(task)
            agent.run(task)

        system = _system_text(requests[-1])
        assert system.count(SKILLS_PROMPT_HEADER.strip()) == 1

    def test_call_llm_does_not_mutate_the_callers_messages(
        self, agent
    ):
        agent.handle_skills(task=None)
        messages = [{"role": "user", "content": "hi"}]
        requests, stub = _capture_requests()

        with stub:
            agent.call_llm(task=None, messages=messages)

        assert messages == [{"role": "user", "content": "hi"}]
        assert "pdf-processing" in _system_text(requests[-1])

    def test_call_llm_prefixes_a_plain_task(self, agent):
        agent.handle_skills(task=None)
        requests, stub = _capture_requests()

        with stub:
            agent.call_llm(task="Summarise the history.")

        user = _user_text(requests[-1])
        assert user.startswith(SKILLS_PROMPT_HEADER.strip())
        assert user.endswith("Summarise the history.")

    def test_no_skills_leaves_the_request_unchanged(self):
        from swarms import Agent

        agent = Agent(
            agent_name="NoSkillsRequestAgent",
            model_name="gpt-4o-mini",
            max_loops=1,
            persistent_memory=False,
            print_on=False,
        )
        requests, stub = _capture_requests()

        with stub:
            agent.call_llm(task="hello")

        assert _user_text(requests[-1]) == "hello"
        assert SKILLS_PROMPT_HEADER.strip() not in _system_text(
            requests[-1]
        )
