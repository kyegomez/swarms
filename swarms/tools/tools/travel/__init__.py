from registry import register


@register("travel")
def travel():
    from api import build_tool

    return build_tool
