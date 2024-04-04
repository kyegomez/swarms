import inspect


def process_tool_docs(item):
    """
    Process the documentation for a given item.

    Args:
        item: The item to process the documentation for.

    Returns:
        metadata: The processed metadata containing the item's name, documentation, and source code.
    """
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
