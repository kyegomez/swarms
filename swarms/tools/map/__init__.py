from ..registry import register


@register("bing_map")
def bing_map():
    from .bing_map.api import build_tool

    return build_tool


@register("baidu_map")
def baidu_map():
    from .baidu_map.baidu_api import build_tool

    return build_tool
