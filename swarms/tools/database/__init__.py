from ..registry import register


@register("database")
def register_database_tool():
    from .api import build_database_tool

    return build_database_tool
