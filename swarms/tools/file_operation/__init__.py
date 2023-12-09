from ..registry import register


@register("file_operation")
def file_operation():
    from .api import build_tool

    return build_tool
