from ..registry import register


@register("wikipedia")
def wikipedia():
    from .api import build_tool

    return build_tool
