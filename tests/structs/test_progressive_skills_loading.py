"""
Tests for progressive/tiered skill loading (Issue #1400).

Verifies:
1. Tier 1: Only skill name + description in system prompt (no full content)
2. Tier 2: load_full_skill is registered as a tool and returns correct content
3. DynamicSkillsLoader no longer stores full content in metadata
4. End-to-end: handle_skills on a real Agent, tool_struct can execute load_full_skill
"""

import json
import os
import tempfile

import pytest

from swarms.structs.dynamic_skills_loader import DynamicSkillsLoader


SKILL_CONTENT = """\
---
name: test-skill
description: A test skill for unit testing
---
# Full Instructions

This is the full skill content that should NOT appear in the system prompt.
It should only be loaded on demand via load_full_skill.
"""

SKILL2_CONTENT = """\
---
name: another-skill
description: Another test skill for similarity matching
---
# Another Skill Instructions

Detailed instructions for another skill.
"""


@pytest.fixture
def skills_dir():
    """Create a temporary skills directory with test skills."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test-skill
        skill1_dir = os.path.join(tmpdir, "test-skill")
        os.makedirs(skill1_dir)
        with open(
            os.path.join(skill1_dir, "SKILL.md"), "w"
        ) as f:
            f.write(SKILL_CONTENT)

        # Create another-skill
        skill2_dir = os.path.join(tmpdir, "another-skill")
        os.makedirs(skill2_dir)
        with open(
            os.path.join(skill2_dir, "SKILL.md"), "w"
        ) as f:
            f.write(SKILL2_CONTENT)

        yield tmpdir


# ── DynamicSkillsLoader unit tests ──────────────────────────────────


class TestDynamicSkillsLoaderTiered:
    """Test that DynamicSkillsLoader implements tiered loading."""

    def test_metadata_does_not_contain_content(self, skills_dir):
        """Tier 1: metadata should only have name, description, path."""
        loader = DynamicSkillsLoader(skills_dir)
        for skill in loader.skills_metadata:
            assert "name" in skill
            assert "description" in skill
            assert "path" in skill
            assert "content" not in skill

    def test_load_full_skill_content_from_disk(self, skills_dir):
        """Tier 2: load_full_skill_content reads full content from disk."""
        loader = DynamicSkillsLoader(skills_dir)
        content = loader.load_full_skill_content("test-skill")
        assert content is not None
        assert "Full Instructions" in content
        assert "should only be loaded on demand" in content

    def test_load_full_skill_content_not_found(self, skills_dir):
        """load_full_skill_content returns None for unknown skill."""
        loader = DynamicSkillsLoader(skills_dir)
        assert loader.load_full_skill_content("nonexistent") is None

    def test_relevant_skills_no_content(self, skills_dir):
        """Relevant skills returned by similarity should not have content."""
        loader = DynamicSkillsLoader(
            skills_dir, similarity_threshold=0.0
        )
        relevant = loader.load_relevant_skills("test skill")
        assert len(relevant) > 0
        for skill in relevant:
            assert "content" not in skill


# ── Agent unit tests ─────────────────────────────────────────────────


class TestAgentSkillsPrompt:
    """Test that Agent builds Tier 1 prompt without full content."""

    def test_build_skills_prompt_no_content(self):
        """_build_skills_prompt should only include name and description."""
        from swarms.structs.agent import Agent

        agent = Agent.__new__(Agent)
        skills = [
            {
                "name": "test-skill",
                "description": "A test skill",
                "path": "/fake/path",
            },
        ]
        prompt = agent._build_skills_prompt(skills)
        assert "test-skill" in prompt
        assert "A test skill" in prompt
        assert "load_full_skill" in prompt
        # Should NOT contain any full content markers
        assert "Full Instructions" not in prompt

    def test_build_skills_prompt_empty(self):
        """_build_skills_prompt returns empty string for no skills."""
        from swarms.structs.agent import Agent

        agent = Agent.__new__(Agent)
        assert agent._build_skills_prompt([]) == ""


class TestAgentLoadFullSkill:
    """Test that Agent.load_full_skill works as Tier 2 loader."""

    def test_load_full_skill_returns_content(self, skills_dir):
        """load_full_skill reads full content from disk on demand."""
        from swarms.structs.agent import Agent

        agent = Agent.__new__(Agent)
        agent.skills_metadata = []

        for skill_folder in os.listdir(skills_dir):
            skill_path = os.path.join(skills_dir, skill_folder)
            if os.path.isdir(skill_path):
                skill_file = os.path.join(skill_path, "SKILL.md")
                if os.path.exists(skill_file):
                    agent.skills_metadata.append(
                        {
                            "name": skill_folder,
                            "description": "test",
                            "path": skill_file,
                        }
                    )

        content = agent.load_full_skill("test-skill")
        assert content is not None
        assert "Full Instructions" in content
        assert "should only be loaded on demand" in content

    def test_load_full_skill_not_found(self):
        """load_full_skill returns None for unknown skill."""
        from swarms.structs.agent import Agent

        agent = Agent.__new__(Agent)
        agent.skills_metadata = []
        assert agent.load_full_skill("nonexistent") is None


# ── Integration tests ────────────────────────────────────────────────


class TestHandleSkillsIntegration:
    """Test the full handle_skills flow on a real Agent instance."""

    def test_handle_skills_static_loading(self, skills_dir):
        """handle_skills(task=None) loads all skills with Tier 1 only."""
        from swarms.structs.agent import Agent

        agent = Agent(
            agent_name="Skills-Test-Agent",
            skills_dir=skills_dir,
            max_loops=1,
            model_name="gpt-4o-mini",
        )

        # handle_skills is called during run(), but we can call it directly
        initial_prompt = agent.system_prompt
        agent.handle_skills(task=None)

        # Tier 1: system prompt has skill names
        assert "test-skill" in agent.system_prompt
        assert "another-skill" in agent.system_prompt

        # Tier 1: system prompt does NOT have full content
        assert "Full Instructions" not in agent.system_prompt
        assert "should only be loaded on demand" not in agent.system_prompt
        assert "Another Skill Instructions" not in agent.system_prompt

        # Tier 2: load_full_skill is registered as a tool
        tool_names = [
            t.get("function", {}).get("name", "")
            for t in agent.tools_list_dictionary
            if isinstance(t, dict)
        ]
        assert "load_full_skill" in tool_names

    def test_handle_skills_dynamic_loading(self, skills_dir):
        """handle_skills(task=...) loads relevant skills with Tier 1 only."""
        from swarms.structs.agent import Agent

        agent = Agent(
            agent_name="Dynamic-Skills-Test-Agent",
            skills_dir=skills_dir,
            max_loops=1,
            model_name="gpt-4o-mini",
        )

        # Use a task that should match skills (low threshold in DynamicSkillsLoader)
        agent.handle_skills(task="test skill for unit testing")

        # Should have loaded at least one skill
        assert "load_full_skill" in [
            t.get("function", {}).get("name", "")
            for t in (agent.tools_list_dictionary or [])
            if isinstance(t, dict)
        ]

        # System prompt should NOT have full content
        assert "should only be loaded on demand" not in agent.system_prompt

    def test_tool_struct_can_execute_load_full_skill(self, skills_dir):
        """Simulate LLM calling load_full_skill — tool_struct executes it."""
        from swarms.structs.agent import Agent

        agent = Agent(
            agent_name="Tool-Exec-Test-Agent",
            skills_dir=skills_dir,
            max_loops=1,
            model_name="gpt-4o-mini",
        )
        agent.handle_skills(task=None)

        # Simulate what the LLM would return as a tool call
        simulated_llm_response = [
            {
                "type": "function",
                "function": {
                    "name": "load_full_skill",
                    "arguments": json.dumps(
                        {"skill_name": "test-skill"}
                    ),
                },
                "id": "call_test_123",
            }
        ]

        # Execute through tool_struct — this is the real execution path
        result = agent.tool_struct.execute_function_calls_from_api_response(
            simulated_llm_response
        )

        # Result should contain the full skill content
        result_str = str(result)
        assert "Full Instructions" in result_str
        assert "should only be loaded on demand" in result_str

    def test_tool_struct_execute_nonexistent_skill(self, skills_dir):
        """load_full_skill returns None for unknown skill via tool_struct."""
        from swarms.structs.agent import Agent

        agent = Agent(
            agent_name="Tool-Exec-Test-Agent-2",
            skills_dir=skills_dir,
            max_loops=1,
            model_name="gpt-4o-mini",
        )
        agent.handle_skills(task=None)

        simulated_llm_response = [
            {
                "type": "function",
                "function": {
                    "name": "load_full_skill",
                    "arguments": json.dumps(
                        {"skill_name": "nonexistent-skill"}
                    ),
                },
                "id": "call_test_456",
            }
        ]

        result = agent.tool_struct.execute_function_calls_from_api_response(
            simulated_llm_response
        )

        # Should return None (no skill found)
        result_str = str(result)
        assert "None" in result_str

    def test_no_duplicate_tool_registration(self, skills_dir):
        """Calling handle_skills twice should not duplicate the tool."""
        from swarms.structs.agent import Agent

        agent = Agent(
            agent_name="Dedup-Test-Agent",
            skills_dir=skills_dir,
            max_loops=1,
            model_name="gpt-4o-mini",
        )
        agent.handle_skills(task=None)
        agent.handle_skills(task=None)

        tool_names = [
            t.get("function", {}).get("name", "")
            for t in agent.tools_list_dictionary
            if isinstance(t, dict)
        ]
        assert tool_names.count("load_full_skill") == 1
