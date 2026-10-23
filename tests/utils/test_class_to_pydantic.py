from collections import OrderedDict, defaultdict

import pytest

from swarms.utils.class_to_pydantic import (
    class_init_to_pydantic_model,
)


class _Holder:
    def __init__(
        self,
        items: list = [],
        mapping: dict = {},
        tags: set = set(),
        buffer: bytearray = bytearray(b"ab"),
        counts: dict = defaultdict(list),
        ordered: dict = OrderedDict(a=1),
    ):
        pass


@pytest.mark.parametrize(
    "field",
    ["items", "mapping", "tags", "buffer", "counts", "ordered"],
)
def test_mutable_default_is_copied_per_instance(field):
    model = class_init_to_pydantic_model(_Holder)
    first, second = model(), model()
    assert getattr(first, field) == getattr(second, field)
    assert getattr(first, field) is not getattr(second, field)


def test_defaultdict_default_keeps_its_type_and_factory():
    value = class_init_to_pydantic_model(_Holder)().counts
    assert isinstance(value, defaultdict)
    assert value.default_factory is list
