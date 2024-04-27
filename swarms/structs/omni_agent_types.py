from typing import (
    Any,
    Callable,
    Sequence,
    Union,
)
from swarms.models.base_llm import BaseLLM
from swarms.models.base_multimodal_model import BaseMultiModalModel
from swarms.structs.agent import Agent

# Unified type for agent
AgentType = Union[Agent, Callable, Any, BaseLLM, BaseMultiModalModel]

# List of agents
AgentListType = Sequence[AgentType]
