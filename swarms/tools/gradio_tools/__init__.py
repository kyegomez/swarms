from ..registry import register


@register("gradio_tools")
def gradio():
    from .api import build_tool

    return build_tool
