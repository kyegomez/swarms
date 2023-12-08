from ..registry import register


@register("stock")
def stock():
    from .api import build_tool

    return build_tool
