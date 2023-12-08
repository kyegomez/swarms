from ..registry import register


@register("google_scholar")
def google_scholar():
    from .api import build_tool

    return build_tool
