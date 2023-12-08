from ..registry import register


@register("bing_search")
def bing_search():
    from .api import build_tool

    return build_tool
