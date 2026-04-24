"""Data models for GraphWorkflow: nodes, edges, and types."""
from enum import Enum
from typing import Any, Dict, Optional, Union

from swarms.structs.agent import Agent

class NodeType(str, Enum):
    AGENT: Agent = "agent"

class Node:
    """
    Represents a node in a graph workflow. Only agent nodes are supported.
    """
    def __init__(
        self,
        id: str = None,
        type: NodeType = NodeType.AGENT,
        agent: Any = None,
        metadata: Dict[str, Any] = None,
    ):
        self.id = id
        self.type = type
        self.agent = agent
        self.metadata = metadata or {}

        if not self.id:
            if self.type == NodeType.AGENT and self.agent is not None:
                self.id = getattr(self.agent, "agent_name", None)
            if not self.id:
                raise ValueError(
                    "Node id could not be auto-assigned. Please provide an id."
                )

    @classmethod
    def from_agent(cls, agent: Agent, **kwargs: Any) -> "Node":
        return cls(
            type=NodeType.AGENT,
            agent=agent,
            id=getattr(agent, "agent_name", None),
            **kwargs,
        )

class Edge:
    """
    Represents an edge in a graph workflow.
    """
    def __init__(
        self,
        source: str = None,
        target: str = None,
        metadata: Dict[str, Any] = None,
    ):
        self.source = source
        self.target = target
        self.metadata = metadata or {}

    @classmethod
    def from_nodes(
        cls,
        source_node: Union["Node", Agent, str],
        target_node: Union["Node", Agent, str],
        **kwargs: Any,
    ) -> "Edge":
        if isinstance(source_node, Node):
            src = source_node.id
        elif hasattr(source_node, "agent_name"):
            src = getattr(source_node, "agent_name", None)
            if src is None:
                raise ValueError("Source agent does not have an agent_name attribute")
        else:
            src = source_node

        if isinstance(target_node, Node):
            tgt = target_node.id
        elif hasattr(target_node, "agent_name"):
            tgt = getattr(target_node, "agent_name", None)
            if tgt is None:
                raise ValueError("Target agent does not have an agent_name attribute")
        else:
            tgt = target_node

        metadata = kwargs if kwargs else None
        return cls(source=src, target=tgt, metadata=metadata)
