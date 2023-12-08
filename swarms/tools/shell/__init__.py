from ..registry import register


@register("shell")
def shell():
    from .api import build_tool

    return build_tool
