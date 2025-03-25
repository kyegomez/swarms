from typing import Literal

# Literal of output types
OutputType = Literal[
    "all",
    "final",
    "list",
    "dict",
    ".json",
    ".md",
    ".txt",
    ".yaml",
    ".toml",
    "string",
    "str",
]

# Use the OutputType for type annotations
output_type: OutputType
