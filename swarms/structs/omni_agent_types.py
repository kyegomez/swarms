from typing import (
    Any,
    Callable,
    Sequence,
    Union,
)
from swarms.structs.agent import Agent

# Unified type for agent
AgentType = Union[Agent, Callable, Any]

# List of agents
AgentListType = Sequence[AgentType]
