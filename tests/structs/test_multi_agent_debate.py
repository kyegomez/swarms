import pytest
from loguru import logger
from swarms.structs.multi_agent_debates import (
    ExpertPanelDiscussion,
)
from swarms.structs.agent import Agent


def create_function_agent(name: str, system_prompt: str = None):
    if system_prompt is None:
        system_prompt = (
            f"You are {name}. Provide concise and direct responses."
        )

    agent = Agent(
        agent_name=name,
        agent_description=f"Test agent {name}",
        system_prompt=system_prompt,
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
    )
    return agent


@pytest.fixture
def sample_two_agents():
    agent1 = create_function_agent(
        "Agent1", "You are Agent1. Provide concise responses."
    )
    agent2 = create_function_agent(
        "Agent2", "You are Agent2. Provide concise responses."
    )
    return [agent1, agent2]


@pytest.fixture
def sample_three_agents():
    agent1 = create_function_agent("Agent1")
    agent2 = create_function_agent("Agent2")
    agent3 = create_function_agent("Agent3")
    return [agent1, agent2, agent3]


@pytest.fixture
def sample_task():
    return "What is 2+2?"


def test_expert_panel_discussion_initialization(sample_three_agents):
    try:
        moderator = create_function_agent("Moderator")
        assert moderator is not None
        panel = ExpertPanelDiscussion(
            max_rounds=2,
            agents=sample_three_agents,
            moderator=moderator,
            output_type="str-all-except-first",
        )
        assert panel is not None
        assert panel.max_rounds == 2
        assert len(panel.agents) == 3
        assert panel.moderator is not None
        logger.info(
            "ExpertPanelDiscussion initialization test passed"
        )
    except Exception as e:
        logger.error(
            f"Failed to test ExpertPanelDiscussion initialization: {e}"
        )
        raise


def test_expert_panel_discussion_run(
    sample_three_agents, sample_task
):
    try:
        moderator = create_function_agent("Moderator")
        assert moderator is not None
        panel = ExpertPanelDiscussion(
            max_rounds=2,
            agents=sample_three_agents,
            moderator=moderator,
            output_type="str-all-except-first",
        )
        assert panel is not None
        result = panel.run(sample_task)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) >= 0
        logger.info("ExpertPanelDiscussion run test passed")
    except Exception as e:
        logger.error(f"Failed to test ExpertPanelDiscussion run: {e}")
        raise


def test_expert_panel_discussion_insufficient_agents(sample_task):
    try:
        moderator = create_function_agent("Moderator")
        assert moderator is not None
        single_agent = [create_function_agent("Agent1")]
        assert single_agent is not None
        assert len(single_agent) > 0
        assert single_agent[0] is not None
        panel = ExpertPanelDiscussion(
            max_rounds=2,
            agents=single_agent,
            moderator=moderator,
            output_type="str-all-except-first",
        )
        assert panel is not None
        with pytest.raises(
            ValueError, match="At least two expert agents"
        ):
            panel.run(sample_task)
        logger.info(
            "ExpertPanelDiscussion insufficient agents test passed"
        )
    except Exception as e:
        logger.error(
            f"Failed to test ExpertPanelDiscussion insufficient agents: {e}"
        )
        raise


def test_expert_panel_discussion_no_moderator(
    sample_three_agents, sample_task
):
    try:
        panel = ExpertPanelDiscussion(
            max_rounds=2,
            agents=sample_three_agents,
            moderator=None,
            output_type="str-all-except-first",
        )
        with pytest.raises(
            ValueError, match="moderator agent is required"
        ):
            panel.run(sample_task)
        logger.info("ExpertPanelDiscussion no moderator test passed")
    except Exception as e:
        logger.error(
            f"Failed to test ExpertPanelDiscussion no moderator: {e}"
        )
        raise


def test_expert_panel_discussion_output_types(
    sample_three_agents, sample_task
):
    try:
        moderator = create_function_agent("Moderator")
        assert moderator is not None
        assert sample_three_agents is not None
        output_types = ["str-all-except-first", "list", "dict", "str"]
        assert output_types is not None
        for output_type in output_types:
            panel = ExpertPanelDiscussion(
                max_rounds=1,
                agents=sample_three_agents,
                moderator=moderator,
                output_type=output_type,
            )
            assert panel is not None
            result = panel.run(sample_task)
            assert result is not None
            if output_type == "list":
                assert isinstance(result, list)
            elif output_type == "dict":
                assert isinstance(result, (dict, list))
            else:
                assert isinstance(result, str)
        logger.info("ExpertPanelDiscussion output types test passed")
    except Exception as e:
        logger.error(
            f"Failed to test ExpertPanelDiscussion output types: {e}"
        )
        raise
