from typing import List
from swarms.structs.base_structure import BaseStructure
from swarms.tools.py_func_to_openai_func_str import (
    get_openai_function_schema_from_func,
)
from swarms.utils.loguru_logger import logger


class ToolDatasetGenerator(BaseStructure):
    """
    Initialize the ToolDatasetGenerator.

    Args:
        functions (List[callable], optional): List of functions to generate examples from. Defaults to None.
        autosave (bool, optional): Flag to enable autosaving generated examples. Defaults to False.
        output_files (List[str], optional): List of output file paths for autosaving. Defaults to None.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
        self,
        functions: List[callable] = None,
        autosave: bool = False,
        output_files: List[str] = None,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        super(ToolDatasetGenerator, self).__init__(*args, **kwargs)
        self.functions = functions
        self.autosave = autosave
        self.output_files = output_files
        self.verbose = verbose

        if self.verbose is True:
            self.log_tool_metadata()

    def run(self, *args, **kwargs):
        """
        Run the ToolDatasetGenerator.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        try:
            for function in self.functions:
                function_str = get_openai_function_schema_from_func(
                    function
                )
                logger.info(function_str)
                if self.autosave:
                    for file in self.output_files:
                        with open(file, "a") as f:
                            f.write(function_str + "\n")
                # agent_response = agent.run(sources_prompts)
                # return agent_response
        except Exception as e:
            logger.info(f"An error occurred: {str(e)}")

    def log_tool_metadata(self):
        """
        Log the number of tools and their metadata.
        """
        try:
            num_tools = len(self.functions)
            logger.info(f"Number of tools: {num_tools}")
            for i, function in enumerate(self.functions):
                logger.info(f"Tool {i+1} metadata:")
                logger.info(f"Name: {function.__name__}")
        except Exception as e:
            logger.info(f"An error occurred: {str(e)}")
