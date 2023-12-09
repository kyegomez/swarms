from ..registry import register


@register("db_diag")
def register_db_diag_tool():
    from .api import build_db_diag_tool

    return build_db_diag_tool
