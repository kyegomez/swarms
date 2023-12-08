from ..registry import register


@register("hugging_tools")
def hugging_tools():
    from .api import build_tool

    return build_tool
