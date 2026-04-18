"""
Unit tests for HeavySwarm Grok 4.20 Heavy 16-agent mode.

Tests verify agent creation, schema structure, flag behavior,
and mutual exclusion without making API calls.
"""

import pytest

from swarms.structs.heavy_swarm import HeavySwarm
from swarms.prompts.heavy_swarm_prompts import (
    grok_heavy_schema,
    GROK_HEAVY_CAPTAIN_PROMPT,
    HARPER_HEAVY_PROMPT,
    BENJAMIN_HEAVY_PROMPT,
    LUCAS_HEAVY_PROMPT,
    OLIVIA_PROMPT,
    JAMES_PROMPT,
    CHARLOTTE_PROMPT,
    HENRY_PROMPT,
    MIA_PROMPT,
    WILLIAM_PROMPT,
    SEBASTIAN_PROMPT,
    JACK_PROMPT,
    OWEN_PROMPT,
    LUNA_PROMPT,
    ELIZABETH_PROMPT,
    NOAH_PROMPT,
)

MODEL = "gpt-4.1-mini"

HEAVY_WORKER_KEYS = [
    "harper",
    "benjamin",
    "lucas",
    "olivia",
    "james",
    "charlotte",
    "henry",
    "mia",
    "william",
    "sebastian",
    "jack",
    "owen",
    "luna",
    "elizabeth",
    "noah",
]


class TestGrokHeavyFlagBehavior:
    """Verify flag defaults and mutual exclusion."""

    def test_grok_heavy_flag_default_false(self):
        swarm = HeavySwarm(
            worker_model_name=MODEL,
            question_agent_model_name=MODEL,
        )
        assert swarm.use_grok_heavy is False

    def test_grok_heavy_flag_set_true(self):
        swarm = HeavySwarm(
            worker_model_name=MODEL,
            question_agent_model_name=MODEL,
            use_grok_heavy=True,
        )
        assert swarm.use_grok_heavy is True

    def test_mutual_exclusion_raises(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            HeavySwarm(
                worker_model_name=MODEL,
                question_agent_model_name=MODEL,
                use_grok_agents=True,
                use_grok_heavy=True,
            )

    def test_grok_agents_still_works(self):
        swarm = HeavySwarm(
            worker_model_name=MODEL,
            question_agent_model_name=MODEL,
            use_grok_agents=True,
        )
        assert "captain" in swarm.agents
        assert "harper" in swarm.agents
        assert len(swarm.agents) == 4

    def test_default_mode_still_works(self):
        swarm = HeavySwarm(
            worker_model_name=MODEL,
            question_agent_model_name=MODEL,
        )
        assert "research" in swarm.agents
        assert len(swarm.agents) == 5


class TestGrokHeavyAgentCreation:
    """Verify 16-agent creation and configuration."""

    @pytest.fixture
    def heavy_swarm(self):
        return HeavySwarm(
            worker_model_name=MODEL,
            question_agent_model_name=MODEL,
            use_grok_heavy=True,
        )

    def test_creates_16_agents(self, heavy_swarm):
        agents = heavy_swarm.agents
        assert len(agents) == 16

    def test_has_captain(self, heavy_swarm):
        assert "captain" in heavy_swarm.agents

    def test_has_all_worker_keys(self, heavy_swarm):
        for key in HEAVY_WORKER_KEYS:
            assert key in heavy_swarm.agents, f"Missing agent: {key}"

    def test_captain_max_loops_is_1(self, heavy_swarm):
        assert heavy_swarm.agents["captain"].max_loops == 1

    def test_captain_print_on_true(self, heavy_swarm):
        assert heavy_swarm.agents["captain"].print_on is True

    def test_captain_name_is_grok(self, heavy_swarm):
        assert heavy_swarm.agents["captain"].agent_name == "Grok"

    def test_worker_agents_use_configured_model(self, heavy_swarm):
        for key in HEAVY_WORKER_KEYS:
            assert heavy_swarm.agents[key].model_name == MODEL

    def test_no_standard_keys_present(self, heavy_swarm):
        for key in [
            "research",
            "analysis",
            "alternatives",
            "verification",
            "synthesis",
        ]:
            assert key not in heavy_swarm.agents


class TestGrokHeavySchema:
    """Verify the grok_heavy_schema structure."""

    def test_schema_is_list(self):
        assert isinstance(grok_heavy_schema, list)
        assert len(grok_heavy_schema) == 1

    def test_function_name(self):
        func = grok_heavy_schema[0]["function"]
        assert func["name"] == "generate_grok_heavy_questions"

    def test_has_16_required_fields(self):
        required = grok_heavy_schema[0]["function"]["parameters"][
            "required"
        ]
        assert len(required) == 16  # thinking + 15 questions

    def test_thinking_is_required(self):
        required = grok_heavy_schema[0]["function"]["parameters"][
            "required"
        ]
        assert "thinking" in required

    def test_all_worker_questions_required(self):
        required = grok_heavy_schema[0]["function"]["parameters"][
            "required"
        ]
        for key in HEAVY_WORKER_KEYS:
            assert (
                f"{key}_question" in required
            ), f"Missing: {key}_question"

    def test_all_properties_have_descriptions(self):
        props = grok_heavy_schema[0]["function"]["parameters"][
            "properties"
        ]
        for field_name, field_def in props.items():
            assert (
                "description" in field_def
            ), f"Missing description for {field_name}"


class TestGrokHeavyPrompts:
    """Verify all prompt constants are non-empty strings."""

    @pytest.mark.parametrize(
        "prompt",
        [
            GROK_HEAVY_CAPTAIN_PROMPT,
            HARPER_HEAVY_PROMPT,
            BENJAMIN_HEAVY_PROMPT,
            LUCAS_HEAVY_PROMPT,
            OLIVIA_PROMPT,
            JAMES_PROMPT,
            CHARLOTTE_PROMPT,
            HENRY_PROMPT,
            MIA_PROMPT,
            WILLIAM_PROMPT,
            SEBASTIAN_PROMPT,
            JACK_PROMPT,
            OWEN_PROMPT,
            LUNA_PROMPT,
            ELIZABETH_PROMPT,
            NOAH_PROMPT,
        ],
    )
    def test_prompt_is_nonempty_string(self, prompt):
        assert isinstance(prompt, str)
        assert len(prompt.strip()) > 100
