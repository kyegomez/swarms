from ..registry import register


@register("douban-film")
def douban_film():
    from .douban import build_tool

    return build_tool
