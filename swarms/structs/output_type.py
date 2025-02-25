from typing import Literal

# Define the output_type using Literal
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
    "str",
]

# Use the OutputType for type annotations
output_type: OutputType
