"""
Pydantic schemas for agent handoffs functionality.

This module defines the data models used for handoffs between agents.
"""

from typing import List
from pydantic import BaseModel, Field


class HandoffRequest(BaseModel):
    """
    Schema for a single handoff request to an agent.

    This model represents a request to delegate a task to a specific agent,
    including the reasoning for the delegation and any task modifications.
    """

    agent_name: str = Field(
        ...,
        description="The name of the agent to delegate the task to. Must exactly match one of the available agent names.",
    )
    task: str = Field(
        ...,
        description="The task to be delegated to the agent. This can be the original task or a modified version optimized for the receiving agent.",
    )
    reasoning: str = Field(
        ...,
        description="Detailed explanation of why this agent was selected and how their capabilities match the task requirements.",
    )


class HandoffTaskSchema(BaseModel):
    """
    Schema for the handoff_task tool.

    This model defines the structure for delegating tasks to one or more agents.
    The agent can use this tool to dynamically send tasks to specialized agents.
    """

    handoffs: List[HandoffRequest] = Field(
        ...,
        min_items=1,
        description="List of handoff requests. Each request specifies an agent and the task to delegate to them. You can delegate to one or multiple agents.",
    )
