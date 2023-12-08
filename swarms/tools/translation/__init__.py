from ..registry import register


@register("nllb-translation")
def translator():
    from .nllb import build_tool

    return build_tool


@register("baidu-translation")
def translator():
    from .baidu import build_tool

    return build_tool
