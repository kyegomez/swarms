from typing import (
    Any,
    Callable,
    Sequence,
    Union,
)
from swarms.models.base_llm import AbstractLLM
from swarms.models.base_multimodal_model import BaseMultiModalModel
from swarms.structs.agent import Agent

# Unified type for agent
AgentType = Union[Agent, Callable, Any, AbstractLLM, BaseMultiModalModel]

# List of agents
AgentListType = Sequence[AgentType]
