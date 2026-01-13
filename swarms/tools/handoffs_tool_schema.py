"""
Handoffs tool schema for agent-to-agent task delegation.

This module provides the tool schema definition for the handoff functionality.
"""

from typing import Any, Dict, List


def get_handoff_tool_schema() -> List[Dict[str, Any]]:
    """
    Get tool definition for handoff functionality.

    Returns:
        List containing the handoff tool definition dictionary.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "handoff_task",
                "description": "Delegate tasks to one or more specialized agents. Use this tool when a task requires specific expertise that another agent possesses, or when you need to break down a complex task into subtasks for different agents. You can delegate to a single agent or multiple agents simultaneously for parallel processing.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "handoffs": {
                            "type": "array",
                            "description": "List of handoff requests. Each request specifies an agent and the task to delegate to them. You can delegate to one or multiple agents.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "agent_name": {
                                        "type": "string",
                                        "description": "The name of the agent to delegate the task to. Must exactly match one of the available agent names.",
                                    },
                                    "task": {
                                        "type": "string",
                                        "description": "The task to be delegated to the agent. This can be the original task or a modified version optimized for the receiving agent's capabilities.",
                                    },
                                    "reasoning": {
                                        "type": "string",
                                        "description": "Detailed explanation of why this agent was selected and how their capabilities match the task requirements. This helps with transparency and decision tracking.",
                                    },
                                },
                                "required": [
                                    "agent_name",
                                    "task",
                                    "reasoning",
                                ],
                            },
                            "minItems": 1,
                        },
                    },
                    "required": ["handoffs"],
                },
            },
        },
    ]
