import inspect
from typing import Callable, Type, Union


def process_tool_docs(item: Union[Callable, Type]) -> str:
    """
    Process the documentation for a given item, which can be a function or a class.

    Args:
        item: The item to process the documentation for. It can be a function or a class.

    Returns:
        str: The processed metadata containing the item's name, documentation, and source code.

    Raises:
        TypeError: If the item is not a function or a class.
    """
    # Check if item is a function or a class
    if not inspect.isfunction(item) and not inspect.isclass(item):
        raise TypeError("Item must be a function or a class.")

    # If item is an instance of a class, get its class
    if not inspect.isclass(item) and hasattr(item, "__class__"):
        item = item.__class__

    doc = inspect.getdoc(item)
    source = inspect.getsource(item)
    is_class = inspect.isclass(item)
    item_type = "Class Name" if is_class else "Function Name"
    metadata = f"{item_type}: {item.__name__}\n\n"
    if doc:
        metadata += f"Documentation:\n{doc}\n\n"
    metadata += f"\n{source}"
    return metadata
