from ..registry import register


@register("weather")
def weather():
    from .api import build_tool

    return build_tool
