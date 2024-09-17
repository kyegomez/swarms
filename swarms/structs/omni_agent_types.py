from typing import (
    Any,
    Callable,
    Sequence,
    Union,
)
from swarm_models.base_llm import BaseLLM
from swarm_models.base_multimodal_model import BaseMultiModalModel
from swarms.structs.agent import Agent

# Unified type for agent
AgentType = Union[Agent, Callable, Any, BaseLLM, BaseMultiModalModel]

# List of agents
AgentListType = Sequence[AgentType]
