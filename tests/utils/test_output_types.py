from typing import get_args

import pytest

from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.output_types import HistoryOutputType


@pytest.mark.parametrize("output_type", get_args(HistoryOutputType))
def test_every_declared_output_type_is_handled(
    output_type, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    conv = Conversation()
    conv.add("user", "What is 2+2?")
    conv.add("assistant", "4.")

    history_output_formatter(conv, output_type)
