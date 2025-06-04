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
    "xml",
    # "dict-final",
    "dict-all-except-first",
    "str-all-except-first",
    "basemodel",
]

OutputType = HistoryOutputType

output_type: HistoryOutputType  # OutputType now includes 'xml'
