from typing import Any, Dict, Optional, Union


def _rejects_forced_choice(model_name: Optional[str]) -> bool:
    """Check whether LiteLLM's model list says the model cannot be forced to call a tool.

    Tries the full name, then the name with each leading provider segment removed.

    Args:
        model_name: The model the request goes to.

    Returns:
        True only when the model is listed with forced tool use unsupported.
    """
    if not model_name:
        return False

    import litellm

    parts = model_name.split("/")
    for start in range(len(parts)):
        info = litellm.model_cost.get("/".join(parts[start:]))
        if info is not None:
            return info.get("supports_forced_tool_use") is False

    return False


def forced_tool_choice(
    tool: Union[Dict[str, Any], str],
    reasoning_effort: Optional[str] = None,
    reasoning_enabled: Optional[bool] = None,
    model_name: Optional[str] = None,
) -> Union[Dict[str, Any], str]:
    """Build the tool_choice that makes the model call one named tool.

    Anthropic rejects a forced choice while extended thinking is on, and some
    models, such as Claude Sonnet 5.5, reject it always. Those get auto instead.

    Args:
        tool: The tool schema in OpenAI format, or its function name.
        reasoning_effort: The agent's reasoning effort, if any.
        reasoning_enabled: Whether the agent has extended thinking on.
        model_name: The agent's model, checked against LiteLLM's model list.

    Returns:
        A tool_choice naming the tool, or "auto" for a reasoning agent or a model that cannot be forced.
    """
    if (
        reasoning_effort
        or reasoning_enabled
        or _rejects_forced_choice(model_name)
    ):
        return "auto"

    name = tool if isinstance(tool, str) else tool["function"]["name"]

    return {"type": "function", "function": {"name": name}}
