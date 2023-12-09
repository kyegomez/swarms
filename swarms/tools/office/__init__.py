from ..registry import register


@register("office-ppt")
def office_ppt():
    from .ppt import build_tool

    return build_tool
