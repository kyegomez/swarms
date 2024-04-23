from typing import Any, List, Optional

from swarms.structs.base_structure import BaseStructure


class BlocksList(BaseStructure):
    """
    A class representing a list of blocks.

    Args:
        name (str): The name of the blocks list.
        description (str): The description of the blocks list.
        blocks (List[Any]): The list of blocks.
        parent (Optional[Any], optional): The parent of the blocks list. Defaults to None.
        **kwargs: Additional keyword arguments.

    Attributes:
        parent (Optional[Any]): The parent of the blocks list.
        blocks (List[Any]): The list of blocks.

    Methods:
        add(block: Any): Add a block to the list.
        remove(block: Any): Remove a block from the list.
        update(block: Any): Update a block in the list.
        get(index: int): Get a block at the specified index.
        get_all(): Get all blocks in the list.
        get_by_name(name: str): Get blocks by name.
        get_by_type(type: str): Get blocks by type.
        get_by_id(id: str): Get blocks by ID.
        get_by_parent(parent: str): Get blocks by parent.
        get_by_parent_id(parent_id: str): Get blocks by parent ID.
        get_by_parent_name(parent_name: str): Get blocks by parent name.
        get_by_parent_type(parent_type: str): Get blocks by parent type.
        get_by_parent_description(parent_description: str): Get blocks by parent description.


    Examples:
    >>> from swarms.structs.block import Block
    >>> from swarms.structs.blockslist import BlocksList
    >>> block = Block("block", "A block")
    >>> blockslist = BlocksList("blockslist", "A list of blocks", [block])
    >>> blockslist

    """

    def __init__(
        self,
        name: str,
        description: str,
        blocks: List[Any],
        parent: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(name=name, description=description, **kwargs)
        self.name = name
        self.description = description
        self.blocks = blocks
        self.parent = parent

    def add(self, block: Any):
        self.blocks.append(block)

    def remove(self, block: Any):
        self.blocks.remove(block)

    def update(self, block: Any):
        self.blocks[self.blocks.index(block)] = block

    def get(self, index: int):
        return self.blocks[index]

    def get_all(self):
        return self.blocks

    def run_block(self, block: Any, task: str, *args, **kwargs):
        """Run the block for the specified task.

        Args:
            task (str): The task to be performed by the block.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The output of the block.

        Raises:
            Exception: If an error occurs during the execution of the block.
        """
        try:
            return block.run(task, *args, **kwargs)
        except Exception as error:
            print(f"[Error] [Block] {error}")
            raise error

    def get_by_name(self, name: str):
        return [block for block in self.blocks if block.name == name]

    def get_by_type(self, type: str):
        return [block for block in self.blocks if block.type == type]

    def get_by_id(self, id: str):
        return [block for block in self.blocks if block.id == id]

    def get_by_parent(self, parent: str):
        return [block for block in self.blocks if block.parent == parent]

    def get_by_parent_id(self, parent_id: str):
        return [
            block for block in self.blocks if block.parent_id == parent_id
        ]

    def get_by_parent_name(self, parent_name: str):
        return [
            block
            for block in self.blocks
            if block.parent_name == parent_name
        ]

    def get_by_parent_type(self, parent_type: str):
        return [
            block
            for block in self.blocks
            if block.parent_type == parent_type
        ]

    def get_by_parent_description(self, parent_description: str):
        return [
            block
            for block in self.blocks
            if block.parent_description == parent_description
        ]

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, index):
        return self.blocks[index]

    def __setitem__(self, index, value):
        self.blocks[index] = value

    def __delitem__(self, index):
        del self.blocks[index]

    def __iter__(self):
        return iter(self.blocks)

    def __reversed__(self):
        return reversed(self.blocks)

    def __contains__(self, item):
        return item in self.blocks

    def __str__(self):
        return f"{self.name}({self.blocks})"

    def __repr__(self):
        return f"{self.name}({self.blocks})"

    def __eq__(self, other):
        return self.blocks == other.blocks
