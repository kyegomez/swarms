from ..registry import register


@register("meta_analysis")
def meta_analysis():
    from .api import build_tool

    return build_tool
