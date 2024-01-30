from dataclasses import dataclass


@dataclass
class ChunkSeparator:
    value: str
    is_prefix: bool = False
