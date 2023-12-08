from ..registry import register


@register("sceneXplain")
def sceneXplain():
    from .api import build_tool

    return build_tool
