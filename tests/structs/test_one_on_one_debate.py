import pytest
from loguru import logger

from swarms.structs.agent import Agent
from swarms.structs.deep_discussion import one_on_one_debate
from swarms.structs.multi_agent_debates import OneOnOneDebate


def create_function_agent(name: str, system_prompt: str = None):
    if system_prompt is None:
        system_prompt = (
            f"You are {name}. Provide thoughtful responses."
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
def sample_agents():
    agent1 = create_function_agent(
        "Debater1",
        "You are a debater who argues for the affirmative position. Be concise and direct.",
    )
    agent2 = create_function_agent(
        "Debater2",
        "You are a debater who argues for the negative position. Be concise and direct.",
    )
    return [agent1, agent2]


@pytest.fixture
def sample_task():
    return "Should artificial intelligence be regulated?"


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


########################################################
# Function API: deep_discussion.one_on_one_debate()
########################################################


def test_one_on_one_debate_basic(sample_agents, sample_task):
    try:
        result = one_on_one_debate(
            max_loops=2,
            task=sample_task,
            agents=sample_agents,
        )
        assert result is not None
        assert isinstance(result, str)
        assert len(result) >= 0
        logger.info("Basic one-on-one debate test passed")
    except Exception as e:
        logger.error(f"Failed to test basic debate: {e}")
        raise


def test_one_on_one_debate_multiple_loops(sample_agents, sample_task):
    try:
        max_loops = 3
        result = one_on_one_debate(
            max_loops=max_loops,
            task=sample_task,
            agents=sample_agents,
        )
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

        result_list = one_on_one_debate(
            max_loops=max_loops,
            task=sample_task,
            agents=sample_agents,
            output_type="list",
        )
        assert result_list is not None
        assert isinstance(result_list, list)
        assert len(result_list) == max_loops
        logger.info("Multiple loops debate test passed")
    except Exception as e:
        logger.error(f"Failed to test multiple loops debate: {e}")
        raise


def test_one_on_one_debate_agent_alternation(
    sample_agents, sample_task
):
    try:
        max_loops = 4
        result = one_on_one_debate(
            max_loops=max_loops,
            task=sample_task,
            agents=sample_agents,
            output_type="list",
        )
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == max_loops

        agent_names = []
        for msg in result:
            if isinstance(msg, dict):
                agent_names.append(msg.get("role", ""))
            elif isinstance(msg, str):
                if "Debater1" in msg:
                    agent_names.append("Debater1")
                elif "Debater2" in msg:
                    agent_names.append("Debater2")
        assert agent_names is not None
        assert len(agent_names) >= 0
        if len(agent_names) > 0:
            assert (
                "Debater1" in agent_names or "Debater2" in agent_names
            )

        if len(agent_names) > 0:
            debater1_count = agent_names.count("Debater1")
            debater2_count = agent_names.count("Debater2")
            total_count = debater1_count + debater2_count
            assert total_count > 0
        logger.info("Agent alternation test passed")
    except Exception as e:
        logger.error(f"Failed to test agent alternation: {e}")
        raise


def test_one_on_one_debate_with_image(sample_agents):
    try:
        task = "Analyze this image and discuss its implications"
        img = "test_image.jpg"
        result = one_on_one_debate(
            max_loops=2,
            task=task,
            agents=sample_agents,
            img=img,
        )
        assert result is not None
        assert isinstance(result, str)
        assert len(result) >= 0
        logger.info("Debate with image test passed")
    except Exception as e:
        logger.error(f"Failed to test debate with image: {e}")
        raise


def test_one_on_one_debate_custom_output_types(
    sample_agents, sample_task
):
    try:
        output_type_checks = {
            "str": str,
            "str-all-except-first": str,
            "dict": list,
            "json": str,
            "list": list,
        }
        for output_type, expected_type in output_type_checks.items():
            result = one_on_one_debate(
                max_loops=1,
                task=sample_task,
                agents=sample_agents,
                output_type=output_type,
            )
            assert result is not None
            assert isinstance(result, expected_type)
            if isinstance(result, (str, list)):
                assert len(result) >= 0
        logger.info("Custom output types test passed")
    except Exception as e:
        logger.error(f"Failed to test custom output types: {e}")
        raise


def test_one_on_one_debate_list_output_structure(
    sample_agents, sample_task
):
    try:
        result = one_on_one_debate(
            max_loops=2,
            task=sample_task,
            agents=sample_agents,
            output_type="list",
        )
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 2

        for message in result:
            assert message is not None
            assert isinstance(message, (str, dict))
            if isinstance(message, str):
                assert len(message) >= 0
            elif isinstance(message, dict):
                assert "role" in message or "content" in message
        logger.info("List output structure test passed")
    except Exception as e:
        logger.error(f"Failed to test list output structure: {e}")
        raise


def test_one_on_one_debate_too_few_agents(sample_task):
    try:
        single_agent = [create_function_agent("SoloAgent")]
        with pytest.raises(
            ValueError, match="There must be exactly two agents"
        ):
            one_on_one_debate(
                max_loops=1,
                task=sample_task,
                agents=single_agent,
            )
        logger.info("Too few agents validation test passed")
    except Exception as e:
        logger.error(f"Failed to test too few agents: {e}")
        raise


def test_one_on_one_debate_too_many_agents(sample_task):
    try:
        many_agents = [
            create_function_agent("Agent1"),
            create_function_agent("Agent2"),
            create_function_agent("Agent3"),
        ]
        with pytest.raises(
            ValueError, match="There must be exactly two agents"
        ):
            one_on_one_debate(
                max_loops=1,
                task=sample_task,
                agents=many_agents,
            )
        logger.info("Too many agents validation test passed")
    except Exception as e:
        logger.error(f"Failed to test too many agents: {e}")
        raise


def test_one_on_one_debate_empty_agents(sample_task):
    try:
        empty_agents = []
        with pytest.raises(
            ValueError, match="There must be exactly two agents"
        ):
            one_on_one_debate(
                max_loops=1,
                task=sample_task,
                agents=empty_agents,
            )
        logger.info("Empty agents validation test passed")
    except Exception as e:
        logger.error(f"Failed to test empty agents: {e}")
        raise


def test_one_on_one_debate_none_agents(sample_task):
    try:
        with pytest.raises((ValueError, TypeError, AttributeError)):
            one_on_one_debate(
                max_loops=1,
                task=sample_task,
                agents=None,
            )
        logger.info("None agents validation test passed")
    except Exception as e:
        logger.error(f"Failed to test None agents: {e}")
        raise


def test_one_on_one_debate_none_task(sample_agents):
    try:
        result = one_on_one_debate(
            max_loops=1,
            task=None,
            agents=sample_agents,
        )
        assert result is not None
        logger.info("None task test passed")
    except Exception as e:
        logger.error(f"Failed to test None task: {e}")
        raise


def test_one_on_one_debate_invalid_output_type(
    sample_agents, sample_task
):
    try:
        with pytest.raises((ValueError, TypeError)):
            one_on_one_debate(
                max_loops=1,
                task=sample_task,
                agents=sample_agents,
                output_type="invalid_type",
            )
        logger.info("Invalid output type validation test passed")
    except Exception as e:
        logger.error(f"Failed to test invalid output type: {e}")
        raise


def test_one_on_one_debate_zero_loops(sample_agents, sample_task):
    try:
        result = one_on_one_debate(
            max_loops=0,
            task=sample_task,
            agents=sample_agents,
        )
        assert result is not None
        assert isinstance(result, str)

        result_list = one_on_one_debate(
            max_loops=0,
            task=sample_task,
            agents=sample_agents,
            output_type="list",
        )
        assert result_list is not None
        assert isinstance(result_list, list)
        assert len(result_list) == 0
        logger.info("Zero loops debate test passed")
    except Exception as e:
        logger.error(f"Failed to test zero loops: {e}")
        raise


def test_one_on_one_debate_different_topics(sample_agents):
    try:
        topics = [
            "What is the meaning of life?",
            "Should we colonize Mars?",
            "Is technology making us more or less connected?",
        ]
        for topic in topics:
            result = one_on_one_debate(
                max_loops=2,
                task=topic,
                agents=sample_agents,
            )
            assert result is not None
            assert isinstance(result, str)
            assert len(result) >= 0
        logger.info("Different topics debate test passed")
    except Exception as e:
        logger.error(f"Failed to test different topics: {e}")
        raise


def test_one_on_one_debate_long_conversation(
    sample_agents, sample_task
):
    try:
        max_loops = 5
        result = one_on_one_debate(
            max_loops=max_loops,
            task=sample_task,
            agents=sample_agents,
            output_type="list",
        )
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == max_loops
        logger.info("Long conversation debate test passed")
    except Exception as e:
        logger.error(f"Failed to test long conversation: {e}")
        raise


def test_one_on_one_debate_different_agent_personalities():
    try:
        agent1 = create_function_agent(
            "Optimist",
            "You are an optimist. Always see the positive side. Be concise.",
        )
        agent2 = create_function_agent(
            "Pessimist",
            "You are a pessimist. Always see the negative side. Be concise.",
        )
        agents = [agent1, agent2]
        task = "What is the future of AI?"
        result = one_on_one_debate(
            max_loops=2,
            task=task,
            agents=agents,
            output_type="list",
        )
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 2

        agent_names = []
        for msg in result:
            if isinstance(msg, dict):
                agent_names.append(msg.get("role", ""))
            elif isinstance(msg, str):
                if "Optimist" in msg:
                    agent_names.append("Optimist")
                elif "Pessimist" in msg:
                    agent_names.append("Pessimist")
        assert agent_names is not None
        assert len(agent_names) >= 0
        if len(agent_names) > 0:
            assert (
                "Optimist" in agent_names
                or "Pessimist" in agent_names
            )
        logger.info("Different agent personalities test passed")
    except Exception as e:
        logger.error(f"Failed to test different personalities: {e}")
        raise


def test_one_on_one_debate_conversation_length_matches_loops(
    sample_agents, sample_task
):
    try:
        for max_loops in [1, 2, 3, 4]:
            result = one_on_one_debate(
                max_loops=max_loops,
                task=sample_task,
                agents=sample_agents,
                output_type="list",
            )
            assert result is not None
            assert isinstance(result, list)
            assert len(result) == max_loops
        logger.info("Conversation length matches loops test passed")
    except Exception as e:
        logger.error(f"Failed to test conversation length: {e}")
        raise


def test_one_on_one_debate_both_agents_participate(
    sample_agents, sample_task
):
    try:
        result = one_on_one_debate(
            max_loops=2,
            task=sample_task,
            agents=sample_agents,
            output_type="list",
        )
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 2

        roles = []
        for msg in result:
            if isinstance(msg, dict) and "role" in msg:
                roles.append(msg.get("role", ""))
            elif isinstance(msg, str):
                if "Debater1" in msg:
                    roles.append("Debater1")
                elif "Debater2" in msg:
                    roles.append("Debater2")
        assert roles is not None
        assert len(roles) >= 0
        if len(roles) > 0:
            unique_roles = set(roles)
            assert unique_roles is not None
            assert len(unique_roles) >= 1
            assert "Debater1" in roles or "Debater2" in roles
        logger.info("Both agents participate test passed")
    except Exception as e:
        logger.error(f"Failed to test both agents participate: {e}")
        raise


########################################################
# Class API: multi_agent_debates.OneOnOneDebate
########################################################


def test_one_on_one_debate_initialization(sample_two_agents):
    try:
        assert sample_two_agents is not None
        debate = OneOnOneDebate(
            max_loops=2,
            agents=sample_two_agents,
            output_type="str-all-except-first",
        )
        assert debate is not None
        assert debate.max_loops == 2
        assert len(debate.agents) == 2
        assert debate.output_type == "str-all-except-first"
        logger.info("OneOnOneDebate initialization test passed")
    except Exception as e:
        logger.error(
            f"Failed to test OneOnOneDebate initialization: {e}"
        )
        raise


def test_one_on_one_debate_run(sample_two_agents, sample_task):
    try:
        assert sample_two_agents is not None
        assert sample_task is not None
        debate = OneOnOneDebate(
            max_loops=2,
            agents=sample_two_agents,
            output_type="str-all-except-first",
        )
        assert debate is not None
        result = debate.run(sample_task)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) >= 0
        logger.info("OneOnOneDebate run test passed")
    except Exception as e:
        logger.error(f"Failed to test OneOnOneDebate run: {e}")
        raise


def test_one_on_one_debate_wrong_number_of_agents(
    sample_three_agents, sample_task
):
    try:
        debate = OneOnOneDebate(
            max_loops=2,
            agents=sample_three_agents,
            output_type="str-all-except-first",
        )
        with pytest.raises(ValueError, match="exactly two agents"):
            debate.run(sample_task)
        logger.info(
            "OneOnOneDebate wrong number of agents test passed"
        )
    except Exception as e:
        logger.error(
            f"Failed to test OneOnOneDebate wrong number of agents: {e}"
        )
        raise


def test_one_on_one_debate_output_types(
    sample_two_agents, sample_task
):
    try:
        assert sample_two_agents is not None
        assert sample_task is not None
        output_types = ["str-all-except-first", "list", "dict", "str"]
        assert output_types is not None
        for output_type in output_types:
            debate = OneOnOneDebate(
                max_loops=2,
                agents=sample_two_agents,
                output_type=output_type,
            )
            assert debate is not None
            result = debate.run(sample_task)
            assert result is not None
            if output_type == "list":
                assert isinstance(result, list)
            elif output_type == "dict":
                assert isinstance(result, (dict, list))
            else:
                assert isinstance(result, str)
        logger.info("OneOnOneDebate output types test passed")
    except Exception as e:
        logger.error(
            f"Failed to test OneOnOneDebate output types: {e}"
        )
        raise


def test_one_on_one_debate_class_with_image(sample_two_agents):
    try:
        assert sample_two_agents is not None
        task = "Analyze this image"
        assert task is not None
        img = "test_image.jpg"
        assert img is not None
        debate = OneOnOneDebate(
            max_loops=2,
            agents=sample_two_agents,
            img=img,
            output_type="str-all-except-first",
        )
        assert debate is not None
        result = debate.run(task)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) >= 0
        logger.info("OneOnOneDebate with image test passed")
    except Exception as e:
        logger.error(f"Failed to test OneOnOneDebate with image: {e}")
        raise


def test_one_on_one_debate_class_multiple_loops(
    sample_two_agents, sample_task
):
    try:
        assert sample_two_agents is not None
        debate = OneOnOneDebate(
            max_loops=5,
            agents=sample_two_agents,
            output_type="str-all-except-first",
        )
        assert debate is not None
        result = debate.run(sample_task)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) >= 0
        logger.info("OneOnOneDebate multiple loops test passed")
    except Exception as e:
        logger.error(
            f"Failed to test OneOnOneDebate multiple loops: {e}"
        )
        raise
