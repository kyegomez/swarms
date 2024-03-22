from typing import Any, List, Union

from pydantic import BaseModel

from swarms.tools.tool import BaseTool
from swarms.utils.loguru_logger import logger


class OmniTool(BaseModel):
    """
    A class representing an OmniTool.

    Attributes:
        tools (Union[List[BaseTool], List[BaseModel], List[Any]]): A list of tools.
        verbose (bool): A flag indicating whether to enable verbose mode.

    Methods:
        transform_models_to_tools(): Transforms models to tools.
        __call__(*args, **kwargs): Calls the tools.

    """

    tools: Union[List[BaseTool], List[BaseModel], List[Any]]
    verbose: bool = False

    def transform_models_to_tools(self):
        """
        Transforms models to tools.
        """
        for i, tool in enumerate(self.tools):
            if isinstance(tool, BaseModel):
                tool_json = tool.model_dump_json()
                # Assuming BaseTool has a method to load from json
                self.tools[i] = BaseTool.load_from_json(tool_json)

    def __call__(self, *args, **kwargs):
        """
        Calls the tools.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tuple: A tuple containing the arguments and keyword arguments.

        """
        try:
            self.transform_models_to_tools()
            logger.info(f"Number of tools: {len(self.tools)}")
            try:
                for tool in self.tools:
                    logger.info(f"Running tool: {tool}")
                    tool(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Error occurred while running tools: {e}"
                )
            return args, kwargs

        except Exception as error:
            logger.error(
                f"Error occurred while running tools: {error}"
            )
            return args, kwargs
