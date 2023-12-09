from ..registry import register


@register("walmart")
def walmart():
    from .api import build_tool

    return build_tool
