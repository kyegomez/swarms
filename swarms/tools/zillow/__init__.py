from ..registry import register


@register("zillow")
def zillow():
    from .api import build_tool

    return build_tool
