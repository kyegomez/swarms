from typing import (
    Any,
    Dict,
    Optional,
)

from swarms.structs.base import BaseStructure


class BlocksDict(BaseStructure):
    """
    A class representing a dictionary of blocks.

    Args:
        name (str): The name of the blocks dictionary.
        description (str): The description of the blocks dictionary.
        blocks (Dict[str, Any]): The dictionary of blocks.
        parent (Optional[Any], optional): The parent of the blocks dictionary. Defaults to None.
        **kwargs: Additional keyword arguments.

    Attributes:
        parent (Optional[Any]): The parent of the blocks dictionary.
        blocks (Dict[str, Any]): The dictionary of blocks.

    Methods:
        add(key: str, block: Any): Add a block to the dictionary.
        remove(key: str): Remove a block from the dictionary.
        get(key: str): Get a block from the dictionary.
    """

    def __init__(
        self,
        name: str,
        description: str,
        blocks: Dict[str, Any],
        parent: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(name=name, description=description, **kwargs)
        self.parent = parent
        self.blocks = blocks

    def add(self, key: str, block: Any):
        self.blocks[key] = block

    def remove(self, key: str):
        del self.blocks[key]

    def get(self, key: str):
        return self.blocks.get(key)

    def update(self, key: str, block: Any):
        self.blocks[key] = block

    def keys(self):
        return list(self.blocks.keys())

    def values(self):
        return list(self.blocks.values())

    def items(self):
        return list(self.blocks.items())

    def clear(self):
        self.blocks.clear()
