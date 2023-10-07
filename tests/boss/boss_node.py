import pytest
from unittest.mock import Mock, patch
from swarms.tools.agent_tools import *
from swarms.boss.boss_node import BossNodeInitializer, BossNode


# For initializing BossNodeInitializer in multiple tests
@pytest.fixture
def mock_boss_node_initializer():
    llm = Mock()
    vectorstore = Mock()
    agent_executor = Mock()
    max_iterations = 5

    boss_node_initializer = BossNodeInitializer(
        llm, vectorstore, agent_executor, max_iterations
    )

    return boss_node_initializer


# Test BossNodeInitializer class __init__ method
def test_boss_node_initializer_init(mock_boss_node_initializer):
    with patch("swarms.tools.agent_tools.BabyAGI.from_llm") as mock_from_llm:
        assert isinstance(mock_boss_node_initializer, BossNodeInitializer)
        mock_from_llm.assert_called_once()


# Test initialize_vectorstore method of BossNodeInitializer class
def test_boss_node_initializer_initialize_vectorstore(mock_boss_node_initializer):
    with patch("swarms.tools.agent_tools.OpenAIEmbeddings") as mock_embeddings, patch(
        "swarms.tools.agent_tools.FAISS"
    ) as mock_faiss:
        result = mock_boss_node_initializer.initialize_vectorstore()
        mock_embeddings.assert_called_once()
        mock_faiss.assert_called_once()
        assert result is not None


# Test initialize_llm method of BossNodeInitializer class
def test_boss_node_initializer_initialize_llm(mock_boss_node_initializer):
    with patch("swarms.tools.agent_tools.OpenAI") as mock_llm:
        result = mock_boss_node_initializer.initialize_llm(mock_llm)
        mock_llm.assert_called_once()
        assert result is not None


# Test create_task method of BossNodeInitializer class
@pytest.mark.parametrize("objective", ["valid objective", ""])
def test_boss_node_initializer_create_task(objective, mock_boss_node_initializer):
    if objective == "":
        with pytest.raises(ValueError):
            mock_boss_node_initializer.create_task(objective)
    else:
        assert mock_boss_node_initializer.create_task(objective) == {
            "objective": objective
        }


# Test run method of BossNodeInitializer class
@pytest.mark.parametrize("task", ["valid task", ""])
def test_boss_node_initializer_run(task, mock_boss_node_initializer):
    with patch.object(mock_boss_node_initializer, "baby_agi"):
        if task == "":
            with pytest.raises(ValueError):
                mock_boss_node_initializer.run(task)
        else:
            try:
                mock_boss_node_initializer.run(task)
                mock_boss_node_initializer.baby_agi.assert_called_once_with(task)
            except Exception:
                pytest.fail("Unexpected Error!")


# Test BossNode function
@pytest.mark.parametrize(
    "api_key, objective, llm_class, max_iterations",
    [
        ("valid_key", "valid_objective", OpenAI, 5),
        ("", "valid_objective", OpenAI, 5),
        ("valid_key", "", OpenAI, 5),
        ("valid_key", "valid_objective", "", 5),
        ("valid_key", "valid_objective", OpenAI, 0),
    ],
)
def test_boss_node(api_key, objective, llm_class, max_iterations):
    with patch("os.getenv") as mock_getenv, patch(
        "swarms.tools.agent_tools.PromptTemplate.from_template"
    ) as mock_from_template, patch(
        "swarms.tools.agent_tools.LLMChain"
    ) as mock_llm_chain, patch(
        "swarms.tools.agent_tools.ZeroShotAgent.create_prompt"
    ) as mock_create_prompt, patch(
        "swarms.tools.agent_tools.ZeroShotAgent"
    ) as mock_zero_shot_agent, patch(
        "swarms.tools.agent_tools.AgentExecutor.from_agent_and_tools"
    ) as mock_from_agent_and_tools, patch(
        "swarms.tools.agent_tools.BossNodeInitializer"
    ) as mock_boss_node_initializer, patch.object(
        mock_boss_node_initializer, "create_task"
    ) as mock_create_task, patch.object(
        mock_boss_node_initializer, "run"
    ) as mock_run:
        if api_key == "" or objective == "" or llm_class == "" or max_iterations <= 0:
            with pytest.raises(ValueError):
                BossNode(
                    objective,
                    api_key,
                    vectorstore=None,
                    worker_node=None,
                    llm_class=llm_class,
                    max_iterations=max_iterations,
                    verbose=False,
                )
        else:
            mock_getenv.return_value = "valid_key"
            BossNode(
                objective,
                api_key,
                vectorstore=None,
                worker_node=None,
                llm_class=llm_class,
                max_iterations=max_iterations,
                verbose=False,
            )
            mock_from_template.assert_called_once()
            mock_llm_chain.assert_called_once()
            mock_create_prompt.assert_called_once()
            mock_zero_shot_agent.assert_called_once()
            mock_from_agent_and_tools.assert_called_once()
            mock_boss_node_initializer.assert_called_once()
            mock_create_task.assert_called_once()
            mock_run.assert_called_once()
