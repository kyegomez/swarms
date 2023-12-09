from ..registry import register


@register("google_places")
def google_places():
    from .api import build_tool

    return build_tool
