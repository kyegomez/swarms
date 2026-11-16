from typing import Literal

HistoryOutputType = Literal[
    "list",
    "dict",
    "dictionary",
    "string",
    "str",
    "final",
    "last",
    "json",
    "all",
    "yaml",
    "dict-all-except-first",
    "str-all-except-first",
    "dict-final",
    "list-final",
]

OutputType = HistoryOutputType
