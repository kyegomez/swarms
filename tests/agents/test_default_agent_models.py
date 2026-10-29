import inspect

import pytest

from swarms.agents.agent_judge import AgentJudge
from swarms.agents.flexion_agent import ReflexionAgent
from swarms.agents.gkp_agent import (
    GKPAgent,
    KnowledgeGenerator,
    Reasoner,
)
from swarms.agents.llm_manager import DEFAULT_MODEL_NAME

# OpenAI shut down o1 on October 23, 2026.
RETIRED_DEFAULT_MODEL = "o1"


@pytest.mark.parametrize(
    "agent_class",
    [
        AgentJudge,
        ReflexionAgent,
        KnowledgeGenerator,
        Reasoner,
        GKPAgent,
    ],
)
def test_default_model_is_not_the_retired_o1(agent_class):
    default = (
        inspect.signature(agent_class.__init__)
        .parameters["model_name"]
        .default
    )

    assert default.split("/")[-1] != RETIRED_DEFAULT_MODEL
    assert default == DEFAULT_MODEL_NAME
