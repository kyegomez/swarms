from ..registry import register


@register("wikidata")
def wikidata():
    from .wikidata import build_tool

    return build_tool
