from ..registry import register


@register("chemical-prop")
def chemical_prop():
    from .prop import build_tool

    return build_tool
