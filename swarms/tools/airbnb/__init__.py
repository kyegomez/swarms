from ..registry import register


@register("airbnb")
def airbnb():
    from .api import build_tool

    return build_tool
