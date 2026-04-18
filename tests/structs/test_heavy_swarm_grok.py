"""
End-to-end tests for HeavySwarm Grok 4.20 Heavy agents.

Tests use real API calls (no mocks) to verify the full
Grok agent pipeline: Captain Swarm, Harper, Benjamin, Lucas.
"""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from swarms.structs.heavy_swarm import (
    BENJAMIN_PROMPT,
    CAPTAIN_SWARM_PROMPT,
    HARPER_PROMPT,
    LUCAS_PROMPT,
    HeavySwarm,
    grok_schema,
)
from swarms.structs.swarm_router import SwarmRouter

MODEL = "gpt-4.1-mini"


# ── Unit-level checks (no API calls) ────────────────────────


class TestGrokAgentCreation:
    """Verify agent creation and configuration."""

    def test_grok_agents_flag_default_false(self):
        """use_grok_agents defaults to False."""
        swarm = HeavySwarm(
            worker_model_name=MODEL,
            question_agent_model_name=MODEL,
        )
        assert swarm.use_grok_agents is False

    def test_grok_agents_flag_set_true(self):
        """use_grok_agents can be set to True."""
        swarm = HeavySwarm(
            worker_model_name=MODEL,
            question_agent_model_name=MODEL,
            use_grok_agents=True,
        )
        assert swarm.use_grok_agents is True

    def test_default_agents_keys(self):
        """Default mode creates research/analysis/etc agents."""
        swarm = HeavySwarm(
            worker_model_name=MODEL,
            question_agent_model_name=MODEL,
            use_grok_agents=False,
        )
        agents = swarm.agents
        assert "research" in agents
        assert "analysis" in agents
        assert "alternatives" in agents
        assert "verification" in agents
        assert "synthesis" in agents
        assert "captain" not in agents

    def test_grok_agents_keys(self):
        """Grok mode creates captain/harper/benjamin/lucas."""
        swarm = HeavySwarm(
            worker_model_name=MODEL,
            question_agent_model_name=MODEL,
            use_grok_agents=True,
        )
        agents = swarm.agents
        assert "captain" in agents
        assert "harper" in agents
        assert "benjamin" in agents
        assert "lucas" in agents
        assert "research" not in agents
        assert "synthesis" not in agents

    def test_grok_agent_names(self):
        """Grok agents have correct agent_name attributes."""
        swarm = HeavySwarm(
            worker_model_name=MODEL,
            question_agent_model_name=MODEL,
            use_grok_agents=True,
        )
        assert swarm.agents["captain"].agent_name == "Captain-Swarm"
        assert swarm.agents["harper"].agent_name == "Harper"
        assert swarm.agents["benjamin"].agent_name == "Benjamin"
        assert swarm.agents["lucas"].agent_name == "Lucas"

    def test_grok_agent_prompts(self):
        """Grok agents use the correct system prompts."""
        swarm = HeavySwarm(
            worker_model_name=MODEL,
            question_agent_model_name=MODEL,
            use_grok_agents=True,
        )
        assert (
            swarm.agents["captain"].system_prompt
            == CAPTAIN_SWARM_PROMPT
        )
        assert swarm.agents["harper"].system_prompt == HARPER_PROMPT
        assert (
            swarm.agents["benjamin"].system_prompt == BENJAMIN_PROMPT
        )
        assert swarm.agents["lucas"].system_prompt == LUCAS_PROMPT

    def test_grok_agent_model_names(self):
        """All grok agents use the configured model."""
        swarm = HeavySwarm(
            worker_model_name=MODEL,
            question_agent_model_name=MODEL,
            use_grok_agents=True,
        )
        for key in ["captain", "harper", "benjamin", "lucas"]:
            assert swarm.agents[key].model_name == MODEL

    def test_grok_agent_tools_passed(self):
        """Worker tools are passed to grok agents."""

        def dummy_tool(x: str) -> str:
            """A dummy tool."""
            return x

        swarm = HeavySwarm(
            worker_model_name=MODEL,
            question_agent_model_name=MODEL,
            use_grok_agents=True,
            worker_tools=[dummy_tool],
        )
        for key in ["captain", "harper", "benjamin", "lucas"]:
            assert swarm.agents[key].tools is not None

    def test_grok_agent_count(self):
        """Grok mode creates exactly 4 agents."""
        swarm = HeavySwarm(
            worker_model_name=MODEL,
            question_agent_model_name=MODEL,
            use_grok_agents=True,
        )
        assert len(swarm.agents) == 4

    def test_default_agent_count(self):
        """Default mode creates exactly 5 agents."""
        swarm = HeavySwarm(
            worker_model_name=MODEL,
            question_agent_model_name=MODEL,
            use_grok_agents=False,
        )
        assert len(swarm.agents) == 5


# ── Schema checks ───────────────────────────────────────────


class TestGrokSchema:
    """Verify grok_schema structure."""

    def test_grok_schema_is_list(self):
        assert isinstance(grok_schema, list)
        assert len(grok_schema) == 1

    def test_grok_schema_function_name(self):
        func = grok_schema[0]["function"]
        assert func["name"] == "generate_grok_questions"

    def test_grok_schema_has_three_questions(self):
        props = grok_schema[0]["function"]["parameters"]["properties"]
        assert "harper_question" in props
        assert "benjamin_question" in props
        assert "lucas_question" in props
        assert "thinking" in props

    def test_grok_schema_required_fields(self):
        required = grok_schema[0]["function"]["parameters"][
            "required"
        ]
        assert "thinking" in required
        assert "harper_question" in required
        assert "benjamin_question" in required
        assert "lucas_question" in required

    def test_grok_schema_no_default_fields(self):
        """Grok schema should not have default agent fields."""
        props = grok_schema[0]["function"]["parameters"]["properties"]
        assert "research_question" not in props
        assert "analysis_question" not in props


# ── Prompt checks ────────────────────────────────────────────


class TestGrokPrompts:
    """Verify prompt constants exist and contain key content."""

    def test_captain_prompt_exists(self):
        assert len(CAPTAIN_SWARM_PROMPT) > 100
        assert "Captain Swarm" in CAPTAIN_SWARM_PROMPT
        assert "orchestrat" in CAPTAIN_SWARM_PROMPT.lower()

    def test_harper_prompt_exists(self):
        assert len(HARPER_PROMPT) > 100
        assert "Harper" in HARPER_PROMPT
        assert "research" in HARPER_PROMPT.lower()
        assert "fact" in HARPER_PROMPT.lower()

    def test_benjamin_prompt_exists(self):
        assert len(BENJAMIN_PROMPT) > 100
        assert "Benjamin" in BENJAMIN_PROMPT
        assert "logic" in BENJAMIN_PROMPT.lower()
        assert "math" in BENJAMIN_PROMPT.lower()

    def test_lucas_prompt_exists(self):
        assert len(LUCAS_PROMPT) > 100
        assert "Lucas" in LUCAS_PROMPT
        assert "creative" in LUCAS_PROMPT.lower()
        assert "contrarian" in LUCAS_PROMPT.lower()


# ── SwarmRouter integration ─────────────────────────────────


class TestSwarmRouterGrokIntegration:
    """Verify SwarmRouter passes through grok flag."""

    def test_router_accepts_grok_flag(self):
        router = SwarmRouter(
            name="TestRouter",
            description="Test",
            swarm_type="HeavySwarm",
            heavy_swarm_loops_per_agent=1,
            heavy_swarm_question_agent_model_name=MODEL,
            heavy_swarm_worker_model_name=MODEL,
            heavy_swarm_use_grok_agents=True,
        )
        assert router.heavy_swarm_use_grok_agents is True

    def test_router_grok_flag_default_false(self):
        router = SwarmRouter(
            name="TestRouter",
            description="Test",
            swarm_type="HeavySwarm",
            heavy_swarm_question_agent_model_name=MODEL,
            heavy_swarm_worker_model_name=MODEL,
        )
        assert router.heavy_swarm_use_grok_agents is False


# ── Question generation (real API) ──────────────────────────


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
class TestGrokQuestionGeneration:
    """Test question generation with real API calls."""

    def test_grok_get_questions_only(self):
        """Grok mode generates harper/benjamin/lucas questions."""
        swarm = HeavySwarm(
            worker_model_name=MODEL,
            question_agent_model_name=MODEL,
            use_grok_agents=True,
        )
        questions = swarm.get_questions_only(
            "What are the best strategies for reducing "
            "carbon emissions in urban areas?"
        )
        assert "error" not in questions
        assert "harper_question" in questions
        assert "benjamin_question" in questions
        assert "lucas_question" in questions
        assert len(questions["harper_question"]) > 0
        assert len(questions["benjamin_question"]) > 0
        assert len(questions["lucas_question"]) > 0

    def test_grok_get_questions_as_list(self):
        """Grok mode returns exactly 3 questions as list."""
        swarm = HeavySwarm(
            worker_model_name=MODEL,
            question_agent_model_name=MODEL,
            use_grok_agents=True,
        )
        questions = swarm.get_questions_as_list(
            "How should a startup approach Series A funding?"
        )
        assert len(questions) == 3
        for q in questions:
            assert isinstance(q, str)
            assert len(q) > 0

    def test_default_get_questions_still_works(self):
        """Default mode still generates 4 questions."""
        swarm = HeavySwarm(
            worker_model_name=MODEL,
            question_agent_model_name=MODEL,
            use_grok_agents=False,
        )
        questions = swarm.get_questions_only(
            "What are the trends in AI hardware?"
        )
        assert "error" not in questions
        assert "research_question" in questions
        assert "analysis_question" in questions
        assert "alternatives_question" in questions
        assert "verification_question" in questions


# ── Full pipeline (real API) ─────────────────────────────────


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
class TestGrokFullPipeline:
    """End-to-end test of the full Grok HeavySwarm pipeline."""

    def test_grok_full_run(self):
        """Full grok pipeline: question gen -> 3 agents -> captain synthesis."""
        swarm = HeavySwarm(
            worker_model_name=MODEL,
            question_agent_model_name=MODEL,
            use_grok_agents=True,
            max_loops=1,
            loops_per_agent=1,
            output_type="string",
        )
        result = swarm.run(
            "What are the key considerations for "
            "building a solar farm in Nevada?"
        )
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 100

    def test_grok_full_run_dict_output(self):
        """Grok pipeline returns structured dict output."""
        swarm = HeavySwarm(
            worker_model_name=MODEL,
            question_agent_model_name=MODEL,
            use_grok_agents=True,
            max_loops=1,
            loops_per_agent=1,
            output_type="dict-all-except-first",
        )
        result = swarm.run(
            "What are the pros and cons of remote work?"
        )
        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0
        # Verify captain synthesis is in the output
        roles = [entry.get("role", "") for entry in result]
        assert any("Captain" in r or "Synthesis" in r for r in roles)

    def test_grok_run_with_dashboard(self):
        """Grok pipeline works with dashboard enabled."""
        swarm = HeavySwarm(
            worker_model_name=MODEL,
            question_agent_model_name=MODEL,
            use_grok_agents=True,
            show_dashboard=True,
            max_loops=1,
            loops_per_agent=1,
            output_type="string",
        )
        result = swarm.run(
            "Compare Python and Rust for systems " "programming"
        )
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 100

    def test_default_full_run_still_works(self):
        """Default pipeline still works after grok changes."""
        swarm = HeavySwarm(
            worker_model_name=MODEL,
            question_agent_model_name=MODEL,
            use_grok_agents=False,
            max_loops=1,
            loops_per_agent=1,
            output_type="string",
        )
        result = swarm.run(
            "What is the current state of quantum " "computing?"
        )
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 100
