from ..registry import register


@register("tutorial")
def tutorial():
    from .api import build_tool

    return build_tool
