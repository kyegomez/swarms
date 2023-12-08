from ..registry import register


@register("code_interpreter")
def code_interpreter():
    from .api import build_tool

    return build_tool
