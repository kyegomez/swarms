from typing import Any, Dict, Optional, Union


def forced_tool_choice(
    tool: Union[Dict[str, Any], str],
    reasoning_effort: Optional[str] = None,
    reasoning_enabled: Optional[bool] = None,
) -> Union[Dict[str, Any], str]:
    """Build the tool_choice that makes the model call one named tool.

    Anthropic rejects a forced choice while extended thinking is on, so an
    agent with reasoning set gets auto instead.

    Args:
        tool: The tool schema in OpenAI format, or its function name.
        reasoning_effort: The agent's reasoning effort, if any.
        reasoning_enabled: Whether the agent has extended thinking on.

    Returns:
        A tool_choice naming the tool, or "auto" for a reasoning agent.
    """
    if reasoning_effort or reasoning_enabled:
        return "auto"

    name = tool if isinstance(tool, str) else tool["function"]["name"]

    return {"type": "function", "function": {"name": name}}
