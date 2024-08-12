from typing import Callable, List, Dict, Any
from loguru import logger


class ToolStorage:
    """
    A class that represents a storage for tools.

    Attributes:
        verbose (bool): A flag to enable verbose logging.
        tools (List[Callable]): A list of tool functions.
        _tools (Dict[str, Callable]): A dictionary that stores the tools, where the key is the tool name and the value is the tool function.
        _settings (Dict[str, Any]): A dictionary that stores the settings, where the key is the setting name and the value is the setting value.
    """

    def __init__(
        self,
        verbose: bool = None,
        tools: List[Callable] = None,
        *args,
        **kwargs,
    ) -> None:
        self.verbose = verbose
        self.tools = tools
        self._tools: Dict[str, Callable] = {}
        self._settings: Dict[str, Any] = {}

    def add_tool(self, func: Callable) -> None:
        """
        Adds a tool to the storage.

        Args:
            func (Callable): The tool function to be added.

        Raises:
            ValueError: If a tool with the same name already exists.
        """
        try:
            logger.info(f"Adding tool: {func.__name__}")
            if func.__name__ in self._tools:
                raise ValueError(
                    f"Tool with name {func.__name__} already exists."
                )
            self._tools[func.__name__] = func
            logger.info(f"Added tool: {func.__name__}")
        except ValueError as e:
            logger.error(e)
            raise

    def get_tool(self, name: str) -> Callable:
        """
        Retrieves a tool by its name.

        Args:
            name (str): The name of the tool to retrieve.

        Returns:
            Callable: The tool function.

        Raises:
            ValueError: If no tool with the given name is found.
        """
        try:
            logger.info(f"Getting tool: {name}")
            if name not in self._tools:
                raise ValueError(f"No tool found with name: {name}")
            return self._tools[name]
        except ValueError as e:
            logger.error(e)
            raise

    def set_setting(self, key: str, value: Any) -> None:
        """
        Sets a setting in the storage.

        Args:
            key (str): The key for the setting.
            value (Any): The value for the setting.
        """
        self._settings[key] = value
        logger.info(f"Setting {key} set to {value}")

    def get_setting(self, key: str) -> Any:
        """
        Gets a setting from the storage.

        Args:
            key (str): The key for the setting.

        Returns:
            Any: The value of the setting.

        Raises:
            KeyError: If the setting is not found.
        """
        try:
            return self._settings[key]
        except KeyError as e:
            logger.error(f"Setting {key} not found error: {e}")
            raise

    def list_tools(self) -> List[str]:
        """
        Lists all registered tools.

        Returns:
            List[str]: A list of tool names.
        """
        return list(self._tools.keys())


# Decorator
def tool_registry(storage: ToolStorage) -> Callable:
    """
    A decorator that registers a function as a tool in the storage.

    Args:
        storage (ToolStorage): The storage instance to register the tool in.

    Returns:
        Callable: The decorator function.
    """

    def decorator(func: Callable) -> Callable:
        logger.info(f"Registering tool: {func.__name__}")
        storage.add_tool(func)

        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                logger.info(f"Tool {func.__name__} executed successfully")
                return result
            except Exception as e:
                logger.error(f"Error executing tool {func.__name__}: {e}")
                raise

        logger.info(f"Registered tool: {func.__name__}")
        return wrapper

    return decorator


# # Test the storage and querying
# if __name__ == "__main__":
#     print(storage.list_tools())  # Should print ['example_tool']
#     # print(use_example_tool(2, 3))  # Should print 5
#     storage.set_setting("example_setting", 42)
#     print(storage.get_setting("example_setting"))  # Should print 42
