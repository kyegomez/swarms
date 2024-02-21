from typing import Any, Dict, Optional

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
        update(key: str, block: Any): Update a block in the dictionary.
        keys(): Get a list of keys in the dictionary.
        values(): Get a list of values in the dictionary.
        items(): Get a list of key-value pairs in the dictionary.
        clear(): Clear all blocks from the dictionary.
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
        """
        Add a block to the dictionary.

        Args:
            key (str): The key of the block.
            block (Any): The block to be added.
        """
        self.blocks[key] = block

    def remove(self, key: str):
        """
        Remove a block from the dictionary.

        Args:
            key (str): The key of the block to be removed.
        """
        del self.blocks[key]

    def get(self, key: str):
        """
        Get a block from the dictionary.

        Args:
            key (str): The key of the block to be retrieved.

        Returns:
            Any: The retrieved block.
        """
        return self.blocks.get(key)

    def update(self, key: str, block: Any):
        """
        Update a block in the dictionary.

        Args:
            key (str): The key of the block to be updated.
            block (Any): The updated block.
        """
        self.blocks[key] = block

    def keys(self):
        """
        Get a list of keys in the dictionary.

        Returns:
            List[str]: A list of keys.
        """
        return list(self.blocks.keys())

    def values(self):
        """
        Get a list of values in the dictionary.

        Returns:
            List[Any]: A list of values.
        """
        return list(self.blocks.values())

    def items(self):
        """
        Get a list of key-value pairs in the dictionary.

        Returns:
            List[Tuple[str, Any]]: A list of key-value pairs.
        """
        return list(self.blocks.items())

    def clear(self):
        """
        Clear all blocks from the dictionary.
        """
        self.blocks.clear()
