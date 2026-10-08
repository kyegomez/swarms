"""Tests for create_agents_from_yaml SwarmConfig / YAML loading."""

from unittest.mock import MagicMock, patch

import pytest

from swarms.agents.create_agents_from_yaml import (
    SwarmConfig,
    create_agents_from_yaml,
    load_yaml_safely,
)

YAML_WITHOUT_DESCRIPTION = """
agents:
  - agent_name: "Researcher"
    system_prompt: "You research topics."
swarm_architecture:
  name: "Research-Swarm"
  swarm_type: "SequentialWorkflow"
  task: "Summarize the topic"
"""

YAML_WITH_DESCRIPTION = """
agents:
  - agent_name: "Researcher"
    system_prompt: "You research topics."
swarm_architecture:
  name: "Research-Swarm"
  description: "A research swarm"
  swarm_type: "SequentialWorkflow"
  task: "Summarize the topic"
"""


def test_swarm_config_description_defaults_when_omitted():
    config = SwarmConfig(
        name="Research-Swarm",
        swarm_type="SequentialWorkflow",
    )
    assert config.description == ""


def test_load_yaml_accepts_swarm_architecture_without_description():
    config = load_yaml_safely(yaml_string=YAML_WITHOUT_DESCRIPTION)
    assert "description" not in config["swarm_architecture"]
    assert config["swarm_architecture"]["name"] == "Research-Swarm"


def test_load_yaml_still_accepts_explicit_description():
    config = load_yaml_safely(yaml_string=YAML_WITH_DESCRIPTION)
    assert (
        config["swarm_architecture"]["description"]
        == "A research swarm"
    )


def test_create_agents_from_yaml_builds_router_without_description():
    fake_agent = MagicMock(name="Researcher")
    fake_router = MagicMock(name="router")

    with (
        patch(
            "swarms.agents.create_agents_from_yaml.create_agent_with_retry",
            return_value=fake_agent,
        ),
        patch(
            "swarms.agents.create_agents_from_yaml.SwarmRouter",
            return_value=fake_router,
        ) as router_cls,
    ):
        result = create_agents_from_yaml(
            yaml_string=YAML_WITHOUT_DESCRIPTION,
            return_type="auto",
        )

    assert result is fake_router
    kwargs = router_cls.call_args.kwargs
    assert kwargs["name"] == "Research-Swarm"
    assert kwargs["description"] == ""
    assert kwargs["swarm_type"] == "SequentialWorkflow"
    assert kwargs["agents"] == [fake_agent]


def test_create_agents_from_yaml_still_requires_name_and_swarm_type():
    yaml_missing_name = """
agents:
  - agent_name: "Researcher"
    system_prompt: "You research topics."
swarm_architecture:
  swarm_type: "SequentialWorkflow"
"""
    with pytest.raises(ValueError, match="name"):
        load_yaml_safely(yaml_string=yaml_missing_name)
