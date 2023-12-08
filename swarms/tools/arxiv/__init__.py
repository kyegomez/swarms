from ..registry import register


@register("arxiv")
def arxiv():
    from .api import build_tool

    return build_tool
