from ..registry import register


@register("python")
def python():
    from .api import build_tool

    return build_tool
