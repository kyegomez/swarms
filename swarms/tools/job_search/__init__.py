from ..registry import register


@register("job_search")
def job_search():
    from .api import build_tool

    return build_tool
