from ..registry import register


@register("google_serper")
def google_serper():
    from .api import build_tool

    return build_tool
