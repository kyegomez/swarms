from swarms.tools.tool_registry import ToolStorage, tool_registry

storage = ToolStorage()


# Example usage
@tool_registry(storage)
def example_tool(x: int, y: int) -> int:
    """
    Example tool function that adds two numbers.

    Args:
        x (int): The first number.
        y (int): The second number.

    Returns:
        int: The sum of the two numbers.
    """
    return x + y


# Query all the tools and get the example tool
print(storage.list_tools())  # Should print ['example_tool']
# print(storage.get_tool('example_tool'))  # Should print <function example_tool at 0x...>

# Find the tool by names and call it
print(storage.get_tool("example_tool"))  # Should print 5


# Test the storage and querying
if __name__ == "__main__":
    print(storage.list_tools())  # Should print ['example_tool']
    print(storage.get_tool("example_tool"))  # Should print 5
    storage.set_setting("example_setting", 42)
    print(storage.get_setting("example_setting"))  # Should print 42
