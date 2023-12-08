from ..registry import register


@register("image_generation")
def image_generation():
    from .api import build_tool

    return build_tool
