from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from swarms.utils.class_to_pydantic import (
    class_init_to_pydantic_model,
)

if TYPE_CHECKING:
    from decimal import Decimal


class _Widget:
    def __init__(self, name: str, price: "Decimal" = None):
        self.name = name


def test_unresolvable_hint_keeps_the_model_usable():
    model = class_init_to_pydantic_model(_Widget)

    assert model(name="bolt").name == "bolt"
    with pytest.raises(ValidationError):
        model(name=3)
