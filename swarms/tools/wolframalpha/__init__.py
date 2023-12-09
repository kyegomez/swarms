from ..registry import register


@register("wolframalpha")
def wolframalpha():
    from .api import build_tool

    return build_tool
