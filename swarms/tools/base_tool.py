import json
from typing import Any, Callable, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field

from swarms.tools.func_to_str import function_to_str, functions_to_str
from swarms.tools.function_util import process_tool_docs
from swarms.tools.py_func_to_openai_func_str import (
    convert_multiple_functions_to_openai_function_schema,
    get_openai_function_schema_from_func,
    load_basemodels_if_needed,
)
from swarms.tools.pydantic_to_json import (
    base_model_to_openai_function,
)
from swarms.tools.tool_parse_exec import parse_and_execute_json
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="base_tool")


# Custom Exceptions
class BaseToolError(Exception):
    """Base exception class for all BaseTool related errors."""

    pass


class ToolValidationError(BaseToolError):
    """Raised when tool validation fails."""

    pass


class ToolExecutionError(BaseToolError):
    """Raised when tool execution fails."""

    pass


class ToolNotFoundError(BaseToolError):
    """Raised when a requested tool is not found."""

    pass


class FunctionSchemaError(BaseToolError):
    """Raised when function schema conversion fails."""

    pass


class ToolDocumentationError(BaseToolError):
    """Raised when tool documentation is missing or invalid."""

    pass


class ToolTypeHintError(BaseToolError):
    """Raised when tool type hints are missing or invalid."""

    pass


ToolType = Union[BaseModel, Dict[str, Any], Callable[..., Any]]


class BaseTool(BaseModel):
    """
    A comprehensive tool management system for function calling, schema conversion, and execution.

    This class provides a unified interface for:
    - Converting functions to OpenAI function calling schemas
    - Managing Pydantic models and their schemas
    - Executing tools with proper error handling and validation
    - Caching expensive operations for improved performance

    Attributes:
        verbose (Optional[bool]): Enable detailed logging output
        base_models (Optional[List[type[BaseModel]]]): List of Pydantic models to manage
        autocheck (Optional[bool]): Enable automatic validation checks
        auto_execute_tool (Optional[bool]): Enable automatic tool execution
        tools (Optional[List[Callable[..., Any]]]): List of callable functions to manage
        tool_system_prompt (Optional[str]): System prompt for tool operations
        function_map (Optional[Dict[str, Callable]]): Mapping of function names to callables
        list_of_dicts (Optional[List[Dict[str, Any]]]): List of dictionary representations

    Examples:
        >>> tool_manager = BaseTool(verbose=True, tools=[my_function])
        >>> schema = tool_manager.func_to_dict(my_function)
        >>> result = tool_manager.execute_tool(response_json)
    """

    verbose: Optional[bool] = None
    base_models: Optional[List[type[BaseModel]]] = None
    autocheck: Optional[bool] = None
    auto_execute_tool: Optional[bool] = None
    tools: Optional[List[Callable[..., Any]]] = None
    tool_system_prompt: Optional[str] = Field(
        None,
        description="The system prompt for the tool system.",
    )
    function_map: Optional[Dict[str, Callable]] = None
    list_of_dicts: Optional[List[Dict[str, Any]]] = None

    def _log_if_verbose(
        self, level: str, message: str, *args, **kwargs
    ) -> None:
        """
        Log message only if verbose mode is enabled.

        Args:
            level (str): Log level ('info', 'error', 'warning', 'debug')
            message (str): Message to log
            *args: Additional arguments for the logger
            **kwargs: Additional keyword arguments for the logger
        """
        if self.verbose:
            log_method = getattr(logger, level.lower(), logger.info)
            log_method(message, *args, **kwargs)

    def _make_hashable(self, obj: Any) -> tuple:
        """
        Convert objects to hashable tuples for caching purposes.

        Args:
            obj: Object to make hashable

        Returns:
            tuple: Hashable representation of the object
        """
        if isinstance(obj, dict):
            return tuple(sorted(obj.items()))
        elif isinstance(obj, list):
            return tuple(obj)
        elif isinstance(obj, type):
            return (obj.__module__, obj.__name__)
        else:
            return obj

    def func_to_dict(
        self,
        function: Callable[..., Any] = None,
    ) -> Dict[str, Any]:
        """
        Convert a callable function to OpenAI function calling schema dictionary.

        This method transforms a Python function into a dictionary format compatible
        with OpenAI's function calling API. Results are cached for performance.

        Args:
            function (Callable[..., Any]): The function to convert
            name (Optional[str]): Override name for the function
            description (str): Override description for the function
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Dict[str, Any]: OpenAI function calling schema dictionary

        Raises:
            FunctionSchemaError: If function schema conversion fails
            ToolValidationError: If function validation fails

        Examples:
            >>> def add(a: int, b: int) -> int:
            ...     '''Add two numbers'''
            ...     return a + b
            >>> tool = BaseTool()
            >>> schema = tool.func_to_dict(add)
        """
        return self.function_to_dict(function)

    def load_params_from_func_for_pybasemodel(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Callable[..., Any]:
        """
        Load and process function parameters for Pydantic BaseModel integration.

        This method prepares function parameters for use with Pydantic BaseModels,
        ensuring proper type handling and validation.

        Args:
            func (Callable[..., Any]): The function to process
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Callable[..., Any]: Processed function with loaded parameters

        Raises:
            ToolValidationError: If function validation fails
            FunctionSchemaError: If parameter loading fails

        Examples:
            >>> tool = BaseTool()
            >>> processed_func = tool.load_params_from_func_for_pybasemodel(my_func)
        """
        if func is None:
            raise ToolValidationError(
                "Function parameter cannot be None"
            )

        try:
            self._log_if_verbose(
                "info",
                f"Loading parameters for function {func.__name__}",
            )

            result = load_basemodels_if_needed(func, *args, **kwargs)

            self._log_if_verbose(
                "info",
                f"Successfully loaded parameters for {func.__name__}",
            )
            return result

        except Exception as e:
            self._log_if_verbose(
                "error",
                f"Failed to load parameters for {func.__name__}: {e}",
            )
            raise FunctionSchemaError(
                f"Failed to load function parameters: {e}"
            ) from e

    def base_model_to_dict(
        self,
        pydantic_type: type[BaseModel],
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Convert a Pydantic BaseModel to OpenAI function calling schema dictionary.

        This method transforms a Pydantic model into a dictionary format compatible
        with OpenAI's function calling API. Results are cached for performance.

        Args:
            pydantic_type (type[BaseModel]): The Pydantic model class to convert
            output_str (bool): Whether to return string output format
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            dict[str, Any]: OpenAI function calling schema dictionary

        Raises:
            ToolValidationError: If pydantic_type validation fails
            FunctionSchemaError: If schema conversion fails

        Examples:
            >>> class MyModel(BaseModel):
            ...     name: str
            ...     age: int
            >>> tool = BaseTool()
            >>> schema = tool.base_model_to_dict(MyModel)
        """
        if pydantic_type is None:
            raise ToolValidationError(
                "Pydantic type parameter cannot be None"
            )

        if not issubclass(pydantic_type, BaseModel):
            raise ToolValidationError(
                "pydantic_type must be a subclass of BaseModel"
            )

        try:
            self._log_if_verbose(
                "info",
                f"Converting Pydantic model {pydantic_type.__name__} to schema",
            )

            # Get the base function schema
            base_result = base_model_to_openai_function(
                pydantic_type, *args, **kwargs
            )

            # Extract the function definition from the functions array
            if (
                "functions" in base_result
                and len(base_result["functions"]) > 0
            ):
                function_def = base_result["functions"][0]

                # Return in proper OpenAI function calling format
                result = {
                    "type": "function",
                    "function": function_def,
                }
            else:
                raise FunctionSchemaError(
                    "Failed to extract function definition from base_model_to_openai_function result"
                )

            self._log_if_verbose(
                "info",
                f"Successfully converted model {pydantic_type.__name__}",
            )
            return result

        except Exception as e:
            self._log_if_verbose(
                "error",
                f"Failed to convert model {pydantic_type.__name__}: {e}",
            )
            raise FunctionSchemaError(
                f"Failed to convert Pydantic model to schema: {e}"
            ) from e

    def multi_base_models_to_dict(
        self, base_models: List[BaseModel]
    ) -> dict[str, Any]:
        """
        Convert multiple Pydantic BaseModels to OpenAI function calling schema.

        This method processes multiple Pydantic models and converts them into
        a unified OpenAI function calling schema format.

        Args:
            return_str (bool): Whether to return string format
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            dict[str, Any]: Combined OpenAI function calling schema

        Raises:
            ToolValidationError: If base_models validation fails
            FunctionSchemaError: If schema conversion fails

        Examples:
            >>> tool = BaseTool(base_models=[Model1, Model2])
            >>> schema = tool.multi_base_models_to_dict()
        """
        if base_models is None:
            raise ToolValidationError(
                "base_models must be set and be a non-empty list before calling this method"
            )

        try:
            return [
                self.base_model_to_dict(model)
                for model in base_models
            ]
        except Exception as e:
            self._log_if_verbose(
                "error", f"Failed to convert multiple models: {e}"
            )
            raise FunctionSchemaError(
                f"Failed to convert multiple Pydantic models: {e}"
            ) from e

    def dict_to_openai_schema_str(
        self,
        dict: dict[str, Any],
    ) -> str:
        """
        Convert a dictionary to OpenAI function calling schema string.

        This method transforms a dictionary representation into a string format
        suitable for OpenAI function calling. Results are cached for performance.

        Args:
            dict (dict[str, Any]): Dictionary to convert

        Returns:
            str: OpenAI schema string representation

        Raises:
            ToolValidationError: If dict validation fails
            FunctionSchemaError: If conversion fails

        Examples:
            >>> tool = BaseTool()
            >>> schema_str = tool.dict_to_openai_schema_str(my_dict)
        """
        if dict is None:
            raise ToolValidationError(
                "Dictionary parameter cannot be None"
            )

        if not isinstance(dict, dict):
            raise ToolValidationError(
                "Parameter must be a dictionary"
            )

        try:
            self._log_if_verbose(
                "info",
                "Converting dictionary to OpenAI schema string",
            )

            result = function_to_str(dict)

            self._log_if_verbose(
                "info",
                "Successfully converted dictionary to schema string",
            )
            return result

        except Exception as e:
            self._log_if_verbose(
                "error",
                f"Failed to convert dictionary to schema string: {e}",
            )
            raise FunctionSchemaError(
                f"Failed to convert dictionary to schema string: {e}"
            ) from e

    def multi_dict_to_openai_schema_str(
        self,
        dicts: list[dict[str, Any]],
    ) -> str:
        """
        Convert multiple dictionaries to OpenAI function calling schema string.

        This method processes a list of dictionaries and converts them into
        a unified OpenAI function calling schema string format.

        Args:
            dicts (list[dict[str, Any]]): List of dictionaries to convert

        Returns:
            str: Combined OpenAI schema string representation

        Raises:
            ToolValidationError: If dicts validation fails
            FunctionSchemaError: If conversion fails

        Examples:
            >>> tool = BaseTool()
            >>> schema_str = tool.multi_dict_to_openai_schema_str([dict1, dict2])
        """
        if dicts is None:
            raise ToolValidationError(
                "Dicts parameter cannot be None"
            )

        if not isinstance(dicts, list) or len(dicts) == 0:
            raise ToolValidationError(
                "Dicts parameter must be a non-empty list"
            )

        for i, d in enumerate(dicts):
            if not isinstance(d, dict):
                raise ToolValidationError(
                    f"Item at index {i} is not a dictionary"
                )

        try:
            self._log_if_verbose(
                "info",
                f"Converting {len(dicts)} dictionaries to schema string",
            )

            result = functions_to_str(dicts)

            self._log_if_verbose(
                "info",
                f"Successfully converted {len(dicts)} dictionaries",
            )
            return result

        except Exception as e:
            self._log_if_verbose(
                "error",
                f"Failed to convert dictionaries to schema string: {e}",
            )
            raise FunctionSchemaError(
                f"Failed to convert dictionaries to schema string: {e}"
            ) from e

    def get_docs_from_callable(self, item):
        """
        Extract documentation from a callable item.

        This method processes a callable and extracts its documentation
        for use in tool schema generation.

        Args:
            item: The callable item to extract documentation from

        Returns:
            The processed documentation

        Raises:
            ToolValidationError: If item validation fails
            ToolDocumentationError: If documentation extraction fails

        Examples:
            >>> tool = BaseTool()
            >>> docs = tool.get_docs_from_callable(my_function)
        """
        if item is None:
            raise ToolValidationError("Item parameter cannot be None")

        if not callable(item):
            raise ToolValidationError("Item must be callable")

        try:
            self._log_if_verbose(
                "info",
                f"Extracting documentation from {getattr(item, '__name__', 'unnamed callable')}",
            )

            result = process_tool_docs(item)

            self._log_if_verbose(
                "info", "Successfully extracted documentation"
            )
            return result

        except Exception as e:
            self._log_if_verbose(
                "error", f"Failed to extract documentation: {e}"
            )
            raise ToolDocumentationError(
                f"Failed to extract documentation: {e}"
            ) from e

    def execute_tool(
        self,
        response: str,
        *args: Any,
        **kwargs: Any,
    ) -> Callable:
        """
        Execute a tool based on a response string.

        This method parses a JSON response string and executes the corresponding
        tool function with proper error handling and validation.

        Args:
            response (str): JSON response string containing tool execution details
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Callable: Result of the tool execution

        Raises:
            ToolValidationError: If response validation fails
            ToolExecutionError: If tool execution fails
            ToolNotFoundError: If specified tool is not found

        Examples:
            >>> tool = BaseTool(tools=[my_function])
            >>> result = tool.execute_tool('{"name": "my_function", "parameters": {...}}')
        """
        if response is None or not isinstance(response, str):
            raise ToolValidationError(
                "Response must be a non-empty string"
            )

        if response.strip() == "":
            raise ToolValidationError("Response cannot be empty")

        if self.tools is None:
            raise ToolValidationError(
                "Tools must be set before executing"
            )

        try:
            self._log_if_verbose(
                "info",
                f"Executing tool with response: {response[:100]}...",
            )

            result = parse_and_execute_json(
                self.tools,
                response,
            )

            self._log_if_verbose(
                "info", "Tool execution completed successfully"
            )
            return result

        except Exception as e:
            self._log_if_verbose(
                "error", f"Tool execution failed: {e}"
            )
            raise ToolExecutionError(
                f"Failed to execute tool: {e}"
            ) from e

    def detect_tool_input_type(self, input: ToolType) -> str:
        """
        Detect the type of tool input for appropriate processing.

        This method analyzes the input and determines whether it's a Pydantic model,
        dictionary, function, or unknown type. Results are cached for performance.

        Args:
            input (ToolType): The input to analyze

        Returns:
            str: Type of the input ("Pydantic", "Dictionary", "Function", or "Unknown")

        Examples:
            >>> tool = BaseTool()
            >>> input_type = tool.detect_tool_input_type(my_function)
            >>> print(input_type)  # "Function"
        """
        if isinstance(input, BaseModel):
            return "Pydantic"
        elif isinstance(input, dict):
            return "Dictionary"
        elif callable(input):
            return "Function"
        else:
            return "Unknown"

    def dynamic_run(self, input: Any) -> str:
        """
        Execute a dynamic run based on the input type with automatic type detection.

        This method automatically detects the input type and processes it accordingly,
        optionally executing the tool if auto_execute_tool is enabled.

        Args:
            input (Any): The input to be processed (Pydantic model, dict, or function)

        Returns:
            str: The result of the dynamic run (schema string or execution result)

        Raises:
            ToolValidationError: If input validation fails
            ToolExecutionError: If auto-execution fails
            FunctionSchemaError: If schema conversion fails

        Examples:
            >>> tool = BaseTool(auto_execute_tool=True)
            >>> result = tool.dynamic_run(my_function)
        """
        if input is None:
            raise ToolValidationError(
                "Input parameter cannot be None"
            )

        try:
            self._log_if_verbose(
                "info",
                "Starting dynamic run with input type detection",
            )

            tool_input_type = self.detect_tool_input_type(input)

            self._log_if_verbose(
                "info", f"Detected input type: {tool_input_type}"
            )

            # Convert input to function schema based on type
            if tool_input_type == "Pydantic":
                function_str = base_model_to_openai_function(input)
            elif tool_input_type == "Dictionary":
                function_str = function_to_str(input)
            elif tool_input_type == "Function":
                function_str = get_openai_function_schema_from_func(
                    input
                )
            else:
                raise ToolValidationError(
                    f"Unknown tool input type: {tool_input_type}"
                )

            # Execute tool if auto-execution is enabled
            if self.auto_execute_tool:
                self._log_if_verbose(
                    "info",
                    "Auto-execution enabled, preparing to execute tool",
                )

                if tool_input_type == "Function":
                    # Initialize tools list if needed
                    if self.tools is None:
                        self.tools = []

                    # Add the function to the tools list if not already present
                    if input not in self.tools:
                        self.tools.append(input)

                # Create or update function map
                if self.function_map is None:
                    self.function_map = {}

                if self.tools:
                    self.function_map.update(
                        {func.__name__: func for func in self.tools}
                    )

                # Execute the tool
                return self.execute_tool(
                    tools=[function_str],
                    function_map=self.function_map,
                )
            else:
                self._log_if_verbose(
                    "info",
                    "Auto-execution disabled, returning schema string",
                )
                return function_str

        except Exception as e:
            self._log_if_verbose("error", f"Dynamic run failed: {e}")
            raise ToolExecutionError(
                f"Dynamic run failed: {e}"
            ) from e

    def execute_tool_by_name(
        self,
        tool_name: str,
        response: str,
    ) -> Any:
        """
        Search for a tool by name and execute it with the provided response.

        This method finds a specific tool in the function map and executes it
        using the provided JSON response string.

        Args:
            tool_name (str): The name of the tool to execute
            response (str): JSON response string containing execution parameters

        Returns:
            Any: The result of executing the tool

        Raises:
            ToolValidationError: If parameters validation fails
            ToolNotFoundError: If the tool with the specified name is not found
            ToolExecutionError: If tool execution fails

        Examples:
            >>> tool = BaseTool(function_map={"add": add_function})
            >>> result = tool.execute_tool_by_name("add", '{"a": 1, "b": 2}')
        """
        if not tool_name or not isinstance(tool_name, str):
            raise ToolValidationError(
                "Tool name must be a non-empty string"
            )

        if not response or not isinstance(response, str):
            raise ToolValidationError(
                "Response must be a non-empty string"
            )

        if self.function_map is None:
            raise ToolValidationError(
                "Function map must be set before executing tools by name"
            )

        try:
            self._log_if_verbose(
                "info", f"Searching for tool: {tool_name}"
            )

            # Find the function in the function map
            func = self.function_map.get(tool_name)

            if func is None:
                raise ToolNotFoundError(
                    f"Tool '{tool_name}' not found in function map"
                )

            self._log_if_verbose(
                "info",
                f"Found tool {tool_name}, executing with response",
            )

            # Execute the tool
            execution = parse_and_execute_json(
                functions=[func],
                json_string=response,
                verbose=self.verbose,
            )

            self._log_if_verbose(
                "info", f"Successfully executed tool {tool_name}"
            )
            return execution

        except ToolNotFoundError:
            raise
        except Exception as e:
            self._log_if_verbose(
                "error", f"Failed to execute tool {tool_name}: {e}"
            )
            raise ToolExecutionError(
                f"Failed to execute tool '{tool_name}': {e}"
            ) from e

    def execute_tool_from_text(self, text: str) -> Any:
        """
        Convert a JSON-formatted string into a tool dictionary and execute the tool.

        This method parses a JSON string representation of a tool call and executes
        the corresponding function with the provided parameters.

        Args:
            text (str): A JSON-formatted string representing a tool call with 'name' and 'parameters' keys

        Returns:
            Any: The result of executing the tool

        Raises:
            ToolValidationError: If text validation fails or JSON parsing fails
            ToolNotFoundError: If the tool with the specified name is not found
            ToolExecutionError: If tool execution fails

        Examples:
            >>> tool = BaseTool(function_map={"add": add_function})
            >>> result = tool.execute_tool_from_text('{"name": "add", "parameters": {"a": 1, "b": 2}}')
        """
        if not text or not isinstance(text, str):
            raise ToolValidationError(
                "Text parameter must be a non-empty string"
            )

        if self.function_map is None:
            raise ToolValidationError(
                "Function map must be set before executing tools from text"
            )

        try:
            self._log_if_verbose(
                "info", f"Parsing tool from text: {text[:100]}..."
            )

            # Convert the text into a dictionary
            try:
                tool = json.loads(text)
            except json.JSONDecodeError as e:
                raise ToolValidationError(
                    f"Invalid JSON format: {e}"
                ) from e

            # Get the tool name and parameters from the dictionary
            tool_name = tool.get("name")
            if not tool_name:
                raise ToolValidationError(
                    "Tool JSON must contain a 'name' field"
                )

            tool_params = tool.get("parameters", {})

            self._log_if_verbose(
                "info", f"Executing tool {tool_name} with parameters"
            )

            # Get the function associated with the tool
            func = self.function_map.get(tool_name)

            # If the function is not found, raise an error
            if func is None:
                raise ToolNotFoundError(
                    f"Tool '{tool_name}' is not mapped to a function"
                )

            # Execute the tool
            result = func(**tool_params)

            self._log_if_verbose(
                "info", f"Successfully executed tool {tool_name}"
            )
            return result

        except (ToolValidationError, ToolNotFoundError):
            raise
        except Exception as e:
            self._log_if_verbose(
                "error", f"Failed to execute tool from text: {e}"
            )
            raise ToolExecutionError(
                f"Failed to execute tool from text: {e}"
            ) from e

    def check_str_for_functions_valid(self, output: str) -> bool:
        """
        Check if the output is a valid JSON string with a function name that matches the function map.

        This method validates that the output string is properly formatted JSON containing
        a function call that exists in the current function map.

        Args:
            output (str): The output string to validate

        Returns:
            bool: True if the output is valid and the function name matches, False otherwise

        Raises:
            ToolValidationError: If output parameter validation fails

        Examples:
            >>> tool = BaseTool(function_map={"add": add_function})
            >>> is_valid = tool.check_str_for_functions_valid('{"type": "function", "function": {"name": "add"}}')
        """
        if not isinstance(output, str):
            raise ToolValidationError("Output must be a string")

        if self.function_map is None:
            self._log_if_verbose(
                "warning",
                "Function map is None, cannot validate function names",
            )
            return False

        try:
            self._log_if_verbose(
                "debug",
                f"Validating output string: {output[:100]}...",
            )

            # Parse the output as JSON
            try:
                data = json.loads(output)
            except json.JSONDecodeError:
                self._log_if_verbose(
                    "debug", "Output is not valid JSON"
                )
                return False

            # Check if the output matches the expected schema
            if (
                data.get("type") == "function"
                and "function" in data
                and "name" in data["function"]
            ):
                # Check if the function name matches any name in the function map
                function_name = data["function"]["name"]
                if function_name in self.function_map:
                    self._log_if_verbose(
                        "debug",
                        f"Valid function call for {function_name}",
                    )
                    return True
                else:
                    self._log_if_verbose(
                        "debug",
                        f"Function {function_name} not found in function map",
                    )
                    return False
            else:
                self._log_if_verbose(
                    "debug",
                    "Output does not match expected function call schema",
                )
                return False

        except Exception as e:
            self._log_if_verbose(
                "error", f"Error validating output: {e}"
            )
            return False

    def convert_funcs_into_tools(self) -> None:
        """
        Convert all functions in the tools list into OpenAI function calling format.

        This method processes all functions in the tools list, validates them for
        proper documentation and type hints, and converts them to OpenAI schemas.
        It also creates a function map for execution.

        Raises:
            ToolValidationError: If tools are not properly configured
            ToolDocumentationError: If functions lack required documentation
            ToolTypeHintError: If functions lack required type hints

        Examples:
            >>> tool = BaseTool(tools=[func1, func2])
            >>> tool.convert_funcs_into_tools()
        """
        if self.tools is None:
            self._log_if_verbose(
                "warning", "No tools provided for conversion"
            )
            return

        if not isinstance(self.tools, list) or len(self.tools) == 0:
            raise ToolValidationError(
                "Tools must be a non-empty list"
            )

        try:
            self._log_if_verbose(
                "info",
                f"Converting {len(self.tools)} functions into tools",
            )
            self._log_if_verbose(
                "info",
                "Ensure functions have documentation and type hints for reliable execution",
            )

            # Transform the tools into OpenAI schema
            schema_result = self.convert_tool_into_openai_schema()

            if schema_result:
                self._log_if_verbose(
                    "info",
                    "Successfully converted tools to OpenAI schema",
                )

            # Create function calling map for all tools
            self.function_map = {
                tool.__name__: tool for tool in self.tools
            }

            self._log_if_verbose(
                "info",
                f"Created function map with {len(self.function_map)} tools",
            )

        except Exception as e:
            self._log_if_verbose(
                "error",
                f"Failed to convert functions into tools: {e}",
            )
            raise ToolValidationError(
                f"Failed to convert functions into tools: {e}"
            ) from e

    def convert_tool_into_openai_schema(self) -> dict[str, Any]:
        """
        Convert tools into OpenAI function calling schema format.

        This method processes all tools and converts them into a unified OpenAI
        function calling schema. Results are cached for performance.

        Returns:
            dict[str, Any]: Combined OpenAI function calling schema

        Raises:
            ToolValidationError: If tools validation fails
            ToolDocumentationError: If tool documentation is missing
            ToolTypeHintError: If tool type hints are missing
            FunctionSchemaError: If schema conversion fails

        Examples:
            >>> tool = BaseTool(tools=[func1, func2])
            >>> schema = tool.convert_tool_into_openai_schema()
        """
        if self.tools is None:
            raise ToolValidationError(
                "Tools must be set before schema conversion"
            )

        if not isinstance(self.tools, list) or len(self.tools) == 0:
            raise ToolValidationError(
                "Tools must be a non-empty list"
            )

        try:
            self._log_if_verbose(
                "info",
                "Converting tools into OpenAI function calling schema",
            )

            tool_schemas = []
            failed_tools = []

            for tool in self.tools:
                try:
                    # Validate tool has documentation and type hints
                    if not self.check_func_if_have_docs(tool):
                        failed_tools.append(
                            f"{tool.__name__} (missing documentation)"
                        )
                        continue

                    if not self.check_func_if_have_type_hints(tool):
                        failed_tools.append(
                            f"{tool.__name__} (missing type hints)"
                        )
                        continue

                    name = tool.__name__
                    description = tool.__doc__

                    self._log_if_verbose(
                        "info", f"Converting tool: {name}"
                    )

                    tool_schema = (
                        get_openai_function_schema_from_func(
                            tool, name=name, description=description
                        )
                    )

                    self._log_if_verbose(
                        "info", f"Tool {name} converted successfully"
                    )
                    tool_schemas.append(tool_schema)

                except Exception as e:
                    failed_tools.append(
                        f"{tool.__name__} (conversion error: {e})"
                    )
                    self._log_if_verbose(
                        "error",
                        f"Failed to convert tool {tool.__name__}: {e}",
                    )

            if failed_tools:
                error_msg = f"Failed to convert tools: {', '.join(failed_tools)}"
                self._log_if_verbose("error", error_msg)
                raise FunctionSchemaError(error_msg)

            if not tool_schemas:
                raise ToolValidationError(
                    "No tools were successfully converted"
                )

            # Combine all tool schemas into a single schema
            combined_schema = {
                "type": "function",
                "functions": [
                    schema["function"] for schema in tool_schemas
                ],
            }

            self._log_if_verbose(
                "info",
                f"Successfully combined {len(tool_schemas)} tool schemas",
            )
            return combined_schema

        except Exception as e:
            if isinstance(
                e, (ToolValidationError, FunctionSchemaError)
            ):
                raise
            self._log_if_verbose(
                "error",
                f"Unexpected error during schema conversion: {e}",
            )
            raise FunctionSchemaError(
                f"Schema conversion failed: {e}"
            ) from e

    def check_func_if_have_docs(self, func: callable) -> bool:
        """
        Check if a function has proper documentation.

        This method validates that a function has a non-empty docstring,
        which is required for reliable tool execution.

        Args:
            func (callable): The function to check

        Returns:
            bool: True if function has documentation

        Raises:
            ToolValidationError: If func is not callable
            ToolDocumentationError: If function lacks documentation

        Examples:
            >>> def documented_func():
            ...     '''This function has docs'''
            ...     pass
            >>> tool = BaseTool()
            >>> has_docs = tool.check_func_if_have_docs(documented_func)  # True
        """
        if not callable(func):
            raise ToolValidationError("Input must be callable")

        if func.__doc__ is not None and func.__doc__.strip():
            self._log_if_verbose(
                "debug", f"Function {func.__name__} has documentation"
            )
            return True
        else:
            error_msg = f"Function {func.__name__} does not have documentation"
            self._log_if_verbose("error", error_msg)
            raise ToolDocumentationError(error_msg)

    def check_func_if_have_type_hints(self, func: callable) -> bool:
        """
        Check if a function has proper type hints.

        This method validates that a function has type annotations,
        which are required for reliable tool execution and schema generation.

        Args:
            func (callable): The function to check

        Returns:
            bool: True if function has type hints

        Raises:
            ToolValidationError: If func is not callable
            ToolTypeHintError: If function lacks type hints

        Examples:
            >>> def typed_func(x: int) -> str:
            ...     '''A typed function'''
            ...     return str(x)
            >>> tool = BaseTool()
            >>> has_hints = tool.check_func_if_have_type_hints(typed_func)  # True
        """
        if not callable(func):
            raise ToolValidationError("Input must be callable")

        if func.__annotations__ and len(func.__annotations__) > 0:
            self._log_if_verbose(
                "debug", f"Function {func.__name__} has type hints"
            )
            return True
        else:
            error_msg = (
                f"Function {func.__name__} does not have type hints"
            )
            self._log_if_verbose("error", error_msg)
            raise ToolTypeHintError(error_msg)

    def find_function_name(
        self, func_name: str
    ) -> Optional[callable]:
        """
        Find a function by name in the tools list.

        This method searches for a function with the specified name
        in the current tools list.

        Args:
            func_name (str): The name of the function to find

        Returns:
            Optional[callable]: The function if found, None otherwise

        Raises:
            ToolValidationError: If func_name is invalid or tools is None

        Examples:
            >>> tool = BaseTool(tools=[my_function])
            >>> func = tool.find_function_name("my_function")
        """
        if not func_name or not isinstance(func_name, str):
            raise ToolValidationError(
                "Function name must be a non-empty string"
            )

        if self.tools is None:
            raise ToolValidationError(
                "Tools must be set before searching for functions"
            )

        self._log_if_verbose(
            "debug", f"Searching for function: {func_name}"
        )

        for func in self.tools:
            if func.__name__ == func_name:
                self._log_if_verbose(
                    "debug", f"Found function: {func_name}"
                )
                return func

        self._log_if_verbose(
            "debug", f"Function {func_name} not found"
        )
        return None

    def function_to_dict(self, func: callable) -> dict:
        """
        Convert a function to dictionary representation.

        This method converts a callable function to its dictionary representation
        using the litellm function_to_dict utility. Results are cached for performance.

        Args:
            func (callable): The function to convert

        Returns:
            dict: Dictionary representation of the function

        Raises:
            ToolValidationError: If func is not callable
            FunctionSchemaError: If conversion fails

        Examples:
            >>> tool = BaseTool()
            >>> func_dict = tool.function_to_dict(my_function)
        """
        if not callable(func):
            raise ToolValidationError("Input must be callable")

        try:
            self._log_if_verbose(
                "debug",
                f"Converting function {func.__name__} to dict",
            )
            result = get_openai_function_schema_from_func(func)
            self._log_if_verbose(
                "debug", f"Successfully converted {func.__name__}"
            )
            return result
        except Exception as e:
            self._log_if_verbose(
                "error",
                f"Failed to convert function {func.__name__} to dict: {e}",
            )
            raise FunctionSchemaError(
                f"Failed to convert function to dict: {e}"
            ) from e

    def multiple_functions_to_dict(
        self, funcs: list[callable]
    ) -> list[dict]:
        """
        Convert multiple functions to dictionary representations.

        This method converts a list of callable functions to their dictionary
        representations using the function_to_dict method.

        Args:
            funcs (list[callable]): List of functions to convert

        Returns:
            list[dict]: List of dictionary representations

        Raises:
            ToolValidationError: If funcs validation fails
            FunctionSchemaError: If any conversion fails

        Examples:
            >>> tool = BaseTool()
            >>> func_dicts = tool.multiple_functions_to_dict([func1, func2])
        """
        if not isinstance(funcs, list):
            raise ToolValidationError("Input must be a list")

        if len(funcs) == 0:
            raise ToolValidationError("Function list cannot be empty")

        for i, func in enumerate(funcs):
            if not callable(func):
                raise ToolValidationError(
                    f"Item at index {i} is not callable"
                )

        try:
            self._log_if_verbose(
                "info",
                f"Converting {len(funcs)} functions to dictionaries",
            )
            result = (
                convert_multiple_functions_to_openai_function_schema(
                    funcs
                )
            )
            self._log_if_verbose(
                "info",
                f"Successfully converted {len(funcs)} functions",
            )
            return result
        except Exception as e:
            self._log_if_verbose(
                "error", f"Failed to convert multiple functions: {e}"
            )
            raise FunctionSchemaError(
                f"Failed to convert multiple functions: {e}"
            ) from e

    def execute_function_with_dict(
        self, func_dict: dict, func_name: Optional[str] = None
    ) -> Any:
        """
        Execute a function using a dictionary of parameters.

        This method executes a function by looking it up by name and passing
        the dictionary as keyword arguments to the function.

        Args:
            func_dict (dict): Dictionary containing function parameters
            func_name (Optional[str]): Name of function to execute (if not in dict)

        Returns:
            Any: Result of function execution

        Raises:
            ToolValidationError: If parameters validation fails
            ToolNotFoundError: If function is not found
            ToolExecutionError: If function execution fails

        Examples:
            >>> tool = BaseTool(tools=[add_function])
            >>> result = tool.execute_function_with_dict({"a": 1, "b": 2}, "add")
        """
        if not isinstance(func_dict, dict):
            raise ToolValidationError(
                "func_dict must be a dictionary"
            )

        try:
            self._log_if_verbose(
                "debug", f"Executing function with dict: {func_dict}"
            )

            # Check if func_name is provided in the dict or as parameter
            if func_name is None:
                func_name = func_dict.get("name") or func_dict.get(
                    "function_name"
                )
                if func_name is None:
                    raise ToolValidationError(
                        "Function name not provided and not found in func_dict"
                    )

            self._log_if_verbose(
                "debug", f"Looking for function: {func_name}"
            )

            # Find the function
            func = self.find_function_name(func_name)
            if func is None:
                raise ToolNotFoundError(
                    f"Function {func_name} not found"
                )

            # Remove function name from parameters before executing
            execution_dict = func_dict.copy()
            execution_dict.pop("name", None)
            execution_dict.pop("function_name", None)

            self._log_if_verbose(
                "debug", f"Executing function {func_name}"
            )
            result = func(**execution_dict)

            self._log_if_verbose(
                "debug", f"Successfully executed {func_name}"
            )
            return result

        except (ToolValidationError, ToolNotFoundError):
            raise
        except Exception as e:
            self._log_if_verbose(
                "error", f"Failed to execute function with dict: {e}"
            )
            raise ToolExecutionError(
                f"Failed to execute function with dict: {e}"
            ) from e

    def execute_multiple_functions_with_dict(
        self,
        func_dicts: list[dict],
        func_names: Optional[list[str]] = None,
    ) -> list[Any]:
        """
        Execute multiple functions using dictionaries of parameters.

        This method executes multiple functions by processing a list of parameter
        dictionaries and optional function names.

        Args:
            func_dicts (list[dict]): List of dictionaries containing function parameters
            func_names (Optional[list[str]]): Optional list of function names

        Returns:
            list[Any]: List of results from function executions

        Raises:
            ToolValidationError: If parameters validation fails
            ToolExecutionError: If any function execution fails

        Examples:
            >>> tool = BaseTool(tools=[add, multiply])
            >>> results = tool.execute_multiple_functions_with_dict([
            ...     {"a": 1, "b": 2}, {"a": 3, "b": 4}
            ... ], ["add", "multiply"])
        """
        if not isinstance(func_dicts, list):
            raise ToolValidationError("func_dicts must be a list")

        if len(func_dicts) == 0:
            raise ToolValidationError("func_dicts cannot be empty")

        if func_names is not None:
            if not isinstance(func_names, list):
                raise ToolValidationError(
                    "func_names must be a list if provided"
                )

            if len(func_names) != len(func_dicts):
                raise ToolValidationError(
                    "func_names length must match func_dicts length"
                )

        try:
            self._log_if_verbose(
                "info",
                f"Executing {len(func_dicts)} functions with dictionaries",
            )

            results = []

            if func_names is None:
                # Execute using names from dictionaries
                for i, func_dict in enumerate(func_dicts):
                    try:
                        result = self.execute_function_with_dict(
                            func_dict
                        )
                        results.append(result)
                    except Exception as e:
                        self._log_if_verbose(
                            "error",
                            f"Failed to execute function at index {i}: {e}",
                        )
                        raise ToolExecutionError(
                            f"Failed to execute function at index {i}: {e}"
                        ) from e
            else:
                # Execute using provided names
                for i, (func_dict, func_name) in enumerate(
                    zip(func_dicts, func_names)
                ):
                    try:
                        result = self.execute_function_with_dict(
                            func_dict, func_name
                        )
                        results.append(result)
                    except Exception as e:
                        self._log_if_verbose(
                            "error",
                            f"Failed to execute function {func_name} at index {i}: {e}",
                        )
                        raise ToolExecutionError(
                            f"Failed to execute function {func_name} at index {i}: {e}"
                        ) from e

            self._log_if_verbose(
                "info",
                f"Successfully executed {len(results)} functions",
            )
            return results

        except ToolExecutionError:
            raise
        except Exception as e:
            self._log_if_verbose(
                "error", f"Failed to execute multiple functions: {e}"
            )
            raise ToolExecutionError(
                f"Failed to execute multiple functions: {e}"
            ) from e

    def validate_function_schema(
        self,
        schema: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]],
        provider: str = "auto",
    ) -> bool:
        """
        Validate the schema of a function for different AI providers.

        This method validates function call schemas for OpenAI, Anthropic, and other providers
        by checking if they conform to the expected structure and contain required fields.

        Args:
            schema: Function schema(s) to validate - can be a single dict or list of dicts
            provider: Target provider format ("openai", "anthropic", "generic", "auto")
                     "auto" attempts to detect the format automatically

        Returns:
            bool: True if schema(s) are valid, False otherwise

        Raises:
            ToolValidationError: If schema parameter is invalid

        Examples:
            >>> tool = BaseTool()
            >>> openai_schema = {
            ...     "type": "function",
            ...     "function": {
            ...         "name": "add_numbers",
            ...         "description": "Add two numbers",
            ...         "parameters": {...}
            ...     }
            ... }
            >>> tool.validate_function_schema(openai_schema, "openai")  # True
        """
        if schema is None:
            self._log_if_verbose(
                "warning", "Schema is None, validation skipped"
            )
            return False

        try:
            # Handle list of schemas
            if isinstance(schema, list):
                if len(schema) == 0:
                    self._log_if_verbose(
                        "warning", "Empty schema list provided"
                    )
                    return False

                # Validate each schema in the list
                for i, single_schema in enumerate(schema):
                    if not self._validate_single_schema(
                        single_schema, provider
                    ):
                        self._log_if_verbose(
                            "error",
                            f"Schema at index {i} failed validation",
                        )
                        return False
                return True

            # Handle single schema
            elif isinstance(schema, dict):
                return self._validate_single_schema(schema, provider)

            else:
                raise ToolValidationError(
                    "Schema must be a dictionary or list of dictionaries"
                )

        except Exception as e:
            self._log_if_verbose(
                "error", f"Schema validation failed: {e}"
            )
            return False

    def _validate_single_schema(
        self, schema: Dict[str, Any], provider: str = "auto"
    ) -> bool:
        """
        Validate a single function schema.

        Args:
            schema: Single function schema dictionary
            provider: Target provider format

        Returns:
            bool: True if schema is valid
        """
        if not isinstance(schema, dict):
            self._log_if_verbose(
                "error", "Schema must be a dictionary"
            )
            return False

        # Auto-detect provider if not specified
        if provider == "auto":
            provider = self._detect_schema_provider(schema)
            self._log_if_verbose(
                "debug", f"Auto-detected provider: {provider}"
            )

        # Validate based on provider
        if provider == "openai":
            return self._validate_openai_schema(schema)
        elif provider == "anthropic":
            return self._validate_anthropic_schema(schema)
        elif provider == "generic":
            return self._validate_generic_schema(schema)
        else:
            self._log_if_verbose(
                "warning",
                f"Unknown provider '{provider}', falling back to generic validation",
            )
            return self._validate_generic_schema(schema)

    def _detect_schema_provider(self, schema: Dict[str, Any]) -> str:
        """
        Auto-detect the provider format of a schema.

        Args:
            schema: Function schema dictionary

        Returns:
            str: Detected provider ("openai", "anthropic", "generic")
        """
        # OpenAI format detection
        if schema.get("type") == "function" and "function" in schema:
            return "openai"

        # Anthropic format detection
        if "input_schema" in schema and "name" in schema:
            return "anthropic"

        # Generic format detection
        if "name" in schema and (
            "parameters" in schema or "arguments" in schema
        ):
            return "generic"

        return "generic"

    def _validate_openai_schema(self, schema: Dict[str, Any]) -> bool:
        """
        Validate OpenAI function calling schema format.

        Expected format:
        {
            "type": "function",
            "function": {
                "name": "function_name",
                "description": "Function description",
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }
        """
        try:
            # Check top-level structure
            if schema.get("type") != "function":
                self._log_if_verbose(
                    "error",
                    "OpenAI schema missing 'type': 'function'",
                )
                return False

            if "function" not in schema:
                self._log_if_verbose(
                    "error", "OpenAI schema missing 'function' key"
                )
                return False

            function_def = schema["function"]
            if not isinstance(function_def, dict):
                self._log_if_verbose(
                    "error", "OpenAI 'function' must be a dictionary"
                )
                return False

            # Check required function fields
            if "name" not in function_def:
                self._log_if_verbose(
                    "error", "OpenAI function missing 'name'"
                )
                return False

            if (
                not isinstance(function_def["name"], str)
                or not function_def["name"].strip()
            ):
                self._log_if_verbose(
                    "error",
                    "OpenAI function 'name' must be a non-empty string",
                )
                return False

            # Description is optional but should be string if present
            if "description" in function_def:
                if not isinstance(function_def["description"], str):
                    self._log_if_verbose(
                        "error",
                        "OpenAI function 'description' must be a string",
                    )
                    return False

            # Validate parameters if present
            if "parameters" in function_def:
                if not self._validate_json_schema(
                    function_def["parameters"]
                ):
                    self._log_if_verbose(
                        "error", "OpenAI function parameters invalid"
                    )
                    return False

            self._log_if_verbose(
                "debug",
                f"OpenAI schema for '{function_def['name']}' is valid",
            )
            return True

        except Exception as e:
            self._log_if_verbose(
                "error", f"OpenAI schema validation error: {e}"
            )
            return False

    def _validate_anthropic_schema(
        self, schema: Dict[str, Any]
    ) -> bool:
        """
        Validate Anthropic tool schema format.

        Expected format:
        {
            "name": "function_name",
            "description": "Function description",
            "input_schema": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        }
        """
        try:
            # Check required fields
            if "name" not in schema:
                self._log_if_verbose(
                    "error", "Anthropic schema missing 'name'"
                )
                return False

            if (
                not isinstance(schema["name"], str)
                or not schema["name"].strip()
            ):
                self._log_if_verbose(
                    "error",
                    "Anthropic 'name' must be a non-empty string",
                )
                return False

            # Description is optional but should be string if present
            if "description" in schema:
                if not isinstance(schema["description"], str):
                    self._log_if_verbose(
                        "error",
                        "Anthropic 'description' must be a string",
                    )
                    return False

            # Validate input_schema if present
            if "input_schema" in schema:
                if not self._validate_json_schema(
                    schema["input_schema"]
                ):
                    self._log_if_verbose(
                        "error", "Anthropic input_schema invalid"
                    )
                    return False

            self._log_if_verbose(
                "debug",
                f"Anthropic schema for '{schema['name']}' is valid",
            )
            return True

        except Exception as e:
            self._log_if_verbose(
                "error", f"Anthropic schema validation error: {e}"
            )
            return False

    def _validate_generic_schema(
        self, schema: Dict[str, Any]
    ) -> bool:
        """
        Validate generic function schema format.

        Expected format (flexible):
        {
            "name": "function_name",
            "description": "Function description" (optional),
            "parameters": {...} or "arguments": {...}
        }
        """
        try:
            # Check required name field
            if "name" not in schema:
                self._log_if_verbose(
                    "error", "Generic schema missing 'name'"
                )
                return False

            if (
                not isinstance(schema["name"], str)
                or not schema["name"].strip()
            ):
                self._log_if_verbose(
                    "error",
                    "Generic 'name' must be a non-empty string",
                )
                return False

            # Description is optional
            if "description" in schema:
                if not isinstance(schema["description"], str):
                    self._log_if_verbose(
                        "error",
                        "Generic 'description' must be a string",
                    )
                    return False

            # Validate parameters or arguments if present
            params_key = None
            if "parameters" in schema:
                params_key = "parameters"
            elif "arguments" in schema:
                params_key = "arguments"

            if params_key:
                if not self._validate_json_schema(schema[params_key]):
                    self._log_if_verbose(
                        "error", f"Generic {params_key} invalid"
                    )
                    return False

            self._log_if_verbose(
                "debug",
                f"Generic schema for '{schema['name']}' is valid",
            )
            return True

        except Exception as e:
            self._log_if_verbose(
                "error", f"Generic schema validation error: {e}"
            )
            return False

    def _validate_json_schema(
        self, json_schema: Dict[str, Any]
    ) -> bool:
        """
        Validate JSON Schema structure for function parameters.

        Args:
            json_schema: JSON Schema dictionary

        Returns:
            bool: True if valid JSON Schema structure
        """
        try:
            if not isinstance(json_schema, dict):
                self._log_if_verbose(
                    "error", "JSON schema must be a dictionary"
                )
                return False

            # Check type field
            if "type" in json_schema:
                valid_types = [
                    "object",
                    "array",
                    "string",
                    "number",
                    "integer",
                    "boolean",
                    "null",
                ]
                if json_schema["type"] not in valid_types:
                    self._log_if_verbose(
                        "error",
                        f"Invalid JSON schema type: {json_schema['type']}",
                    )
                    return False

            # For object type, validate properties
            if json_schema.get("type") == "object":
                if "properties" in json_schema:
                    if not isinstance(
                        json_schema["properties"], dict
                    ):
                        self._log_if_verbose(
                            "error",
                            "JSON schema 'properties' must be a dictionary",
                        )
                        return False

                    # Validate each property
                    for prop_name, prop_def in json_schema[
                        "properties"
                    ].items():
                        if not isinstance(prop_def, dict):
                            self._log_if_verbose(
                                "error",
                                f"Property '{prop_name}' definition must be a dictionary",
                            )
                            return False

                        # Recursively validate nested schemas
                        if not self._validate_json_schema(prop_def):
                            return False

                # Validate required field
                if "required" in json_schema:
                    if not isinstance(json_schema["required"], list):
                        self._log_if_verbose(
                            "error",
                            "JSON schema 'required' must be a list",
                        )
                        return False

                    # Check that required fields exist in properties
                    if "properties" in json_schema:
                        properties = json_schema["properties"]
                        for required_field in json_schema["required"]:
                            if required_field not in properties:
                                self._log_if_verbose(
                                    "error",
                                    f"Required field '{required_field}' not in properties",
                                )
                                return False

            # For array type, validate items
            if json_schema.get("type") == "array":
                if "items" in json_schema:
                    if not self._validate_json_schema(
                        json_schema["items"]
                    ):
                        return False

            return True

        except Exception as e:
            self._log_if_verbose(
                "error", f"JSON schema validation error: {e}"
            )
            return False

    def get_schema_provider_format(
        self, schema: Dict[str, Any]
    ) -> str:
        """
        Get the detected provider format of a schema.

        Args:
            schema: Function schema dictionary

        Returns:
            str: Provider format ("openai", "anthropic", "generic", "unknown")

        Examples:
            >>> tool = BaseTool()
            >>> provider = tool.get_schema_provider_format(my_schema)
            >>> print(provider)  # "openai"
        """
        if not isinstance(schema, dict):
            return "unknown"

        return self._detect_schema_provider(schema)

    def convert_schema_between_providers(
        self, schema: Dict[str, Any], target_provider: str
    ) -> Dict[str, Any]:
        """
        Convert a function schema between different provider formats.

        Args:
            schema: Source function schema
            target_provider: Target provider format ("openai", "anthropic", "generic")

        Returns:
            Dict[str, Any]: Converted schema

        Raises:
            ToolValidationError: If conversion fails

        Examples:
            >>> tool = BaseTool()
            >>> anthropic_schema = tool.convert_schema_between_providers(openai_schema, "anthropic")
        """
        if not isinstance(schema, dict):
            raise ToolValidationError("Schema must be a dictionary")

        source_provider = self._detect_schema_provider(schema)

        if source_provider == target_provider:
            self._log_if_verbose(
                "debug", f"Schema already in {target_provider} format"
            )
            return schema.copy()

        try:
            # Extract common fields
            name = self._extract_function_name(
                schema, source_provider
            )
            description = self._extract_function_description(
                schema, source_provider
            )
            parameters = self._extract_function_parameters(
                schema, source_provider
            )

            # Convert to target format
            if target_provider == "openai":
                return self._build_openai_schema(
                    name, description, parameters
                )
            elif target_provider == "anthropic":
                return self._build_anthropic_schema(
                    name, description, parameters
                )
            elif target_provider == "generic":
                return self._build_generic_schema(
                    name, description, parameters
                )
            else:
                raise ToolValidationError(
                    f"Unknown target provider: {target_provider}"
                )

        except Exception as e:
            self._log_if_verbose(
                "error", f"Schema conversion failed: {e}"
            )
            raise ToolValidationError(
                f"Failed to convert schema: {e}"
            ) from e

    def _extract_function_name(
        self, schema: Dict[str, Any], provider: str
    ) -> str:
        """Extract function name from schema based on provider format."""
        if provider == "openai":
            return schema.get("function", {}).get("name", "")
        else:  # anthropic, generic
            return schema.get("name", "")

    def _extract_function_description(
        self, schema: Dict[str, Any], provider: str
    ) -> Optional[str]:
        """Extract function description from schema based on provider format."""
        if provider == "openai":
            return schema.get("function", {}).get("description")
        else:  # anthropic, generic
            return schema.get("description")

    def _extract_function_parameters(
        self, schema: Dict[str, Any], provider: str
    ) -> Optional[Dict[str, Any]]:
        """Extract function parameters from schema based on provider format."""
        if provider == "openai":
            return schema.get("function", {}).get("parameters")
        elif provider == "anthropic":
            return schema.get("input_schema")
        else:  # generic
            return schema.get("parameters") or schema.get("arguments")

    def _build_openai_schema(
        self,
        name: str,
        description: Optional[str],
        parameters: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build OpenAI format schema."""
        function_def = {"name": name}
        if description:
            function_def["description"] = description
        if parameters:
            function_def["parameters"] = parameters

        return {"type": "function", "function": function_def}

    def _build_anthropic_schema(
        self,
        name: str,
        description: Optional[str],
        parameters: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build Anthropic format schema."""
        schema = {"name": name}
        if description:
            schema["description"] = description
        if parameters:
            schema["input_schema"] = parameters

        return schema

    def _build_generic_schema(
        self,
        name: str,
        description: Optional[str],
        parameters: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build generic format schema."""
        schema = {"name": name}
        if description:
            schema["description"] = description
        if parameters:
            schema["parameters"] = parameters

        return schema

    def execute_function_calls_from_api_response(
        self,
        api_response: Union[Dict[str, Any], str, List[Any]],
        sequential: bool = False,
        max_workers: int = 4,
        return_as_string: bool = True,
    ) -> Union[List[Any], List[str]]:
        """
        Automatically detect and execute function calls from OpenAI or Anthropic API responses.

        This method can handle:
        - OpenAI API responses with tool_calls
        - Anthropic API responses with tool use (including BaseModel objects)
        - Direct list of tool call objects (from OpenAI ChatCompletionMessageToolCall or Anthropic BaseModels)
        - Pydantic BaseModel objects from Anthropic responses
        - Parallel function execution with concurrent.futures or sequential execution
        - Multiple function calls in a single response

        Args:
            api_response (Union[Dict[str, Any], str, List[Any]]): The API response containing function calls
            sequential (bool): If True, execute functions sequentially. If False, execute in parallel (default)
            max_workers (int): Maximum number of worker threads for parallel execution (default: 4)
            return_as_string (bool): If True, return results as formatted strings (default: True)

        Returns:
            Union[List[Any], List[str]]: List of results from executed functions

        Raises:
            ToolValidationError: If API response validation fails
            ToolNotFoundError: If any function is not found
            ToolExecutionError: If function execution fails

        Examples:
            >>> # OpenAI API response example
            >>> openai_response = {
            ...     "choices": [{"message": {"tool_calls": [...]}}]
            ... }
            >>> tool = BaseTool(tools=[weather_function])
            >>> results = tool.execute_function_calls_from_api_response(openai_response)

            >>> # Direct tool calls list (including BaseModel objects)
            >>> tool_calls = [ChatCompletionMessageToolCall(...), ...]
            >>> results = tool.execute_function_calls_from_api_response(tool_calls)
        """
        # Handle None API response gracefully by returning empty results
        if api_response is None:
            self._log_if_verbose(
                "warning",
                "API response is None, returning empty results. This may indicate the LLM did not return a valid response.",
            )
            return [] if not return_as_string else []

        # Handle direct list of tool call objects (e.g., from OpenAI ChatCompletionMessageToolCall or Anthropic BaseModels)
        if isinstance(api_response, list):
            self._log_if_verbose(
                "info",
                "Processing direct list of tool call objects",
            )
            function_calls = (
                self._extract_function_calls_from_tool_call_objects(
                    api_response
                )
            )
        # Handle single BaseModel object (common with Anthropic responses)
        elif isinstance(api_response, BaseModel):
            self._log_if_verbose(
                "info",
                "Processing single BaseModel object (likely Anthropic response)",
            )
            # Convert BaseModel to dict and process
            api_response_dict = api_response.model_dump()
            function_calls = (
                self._extract_function_calls_from_response(
                    api_response_dict
                )
            )
        else:
            # Convert string to dict if needed
            if isinstance(api_response, str):
                try:
                    api_response = json.loads(api_response)
                except json.JSONDecodeError as e:
                    self._log_if_verbose(
                        "error",
                        f"Failed to parse JSON from API response: {e}. Response: '{api_response[:100]}...'",
                    )
                    return []

            if not isinstance(api_response, dict):
                self._log_if_verbose(
                    "warning",
                    f"API response is not a dictionary (type: {type(api_response)}), returning empty list",
                )
                return []

            # Extract function calls from dictionary response
            function_calls = (
                self._extract_function_calls_from_response(
                    api_response
                )
            )

        if self.function_map is None and self.tools is None:
            raise ToolValidationError(
                "Either function_map or tools must be set before executing function calls"
            )

        try:
            if not function_calls:
                self._log_if_verbose(
                    "warning",
                    "No function calls found in API response",
                )
                return []

            self._log_if_verbose(
                "info",
                f"Found {len(function_calls)} function call(s)",
            )

            # Ensure function_map is available
            if self.function_map is None and self.tools is not None:
                self.function_map = {
                    tool.__name__: tool for tool in self.tools
                }

            # Execute function calls
            if sequential:
                results = self._execute_function_calls_sequential(
                    function_calls
                )
            else:
                results = self._execute_function_calls_parallel(
                    function_calls, max_workers
                )

            # Format results as strings if requested
            if return_as_string:
                return self._format_results_as_strings(
                    results, function_calls
                )
            else:
                return results

        except Exception as e:
            self._log_if_verbose(
                "error",
                f"Failed to execute function calls from API response: {e}",
            )
            raise ToolExecutionError(
                f"Failed to execute function calls from API response: {e}"
            ) from e

    def _extract_function_calls_from_response(
        self, response: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract function calls from different API response formats.

        Args:
            response: API response dictionary

        Returns:
            List[Dict[str, Any]]: List of standardized function call dictionaries
        """
        function_calls = []

        # Try OpenAI format first
        openai_calls = self._extract_openai_function_calls(response)
        if openai_calls:
            function_calls.extend(openai_calls)
            self._log_if_verbose(
                "debug",
                f"Extracted {len(openai_calls)} OpenAI function calls",
            )

        # Try Anthropic format
        anthropic_calls = self._extract_anthropic_function_calls(
            response
        )
        if anthropic_calls:
            function_calls.extend(anthropic_calls)
            self._log_if_verbose(
                "debug",
                f"Extracted {len(anthropic_calls)} Anthropic function calls",
            )

        # Try generic format (direct function calls)
        generic_calls = self._extract_generic_function_calls(response)
        if generic_calls:
            function_calls.extend(generic_calls)
            self._log_if_verbose(
                "debug",
                f"Extracted {len(generic_calls)} generic function calls",
            )

        return function_calls

    def _extract_openai_function_calls(
        self, response: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract function calls from OpenAI API response format."""
        function_calls = []

        try:
            # Check if the response itself is a single function call object
            if (
                response.get("type") == "function"
                and "function" in response
            ):
                function_info = response.get("function", {})
                name = function_info.get("name")
                arguments_str = function_info.get("arguments", "{}")

                if name:
                    try:
                        # Parse arguments JSON string
                        arguments = (
                            json.loads(arguments_str)
                            if isinstance(arguments_str, str)
                            else arguments_str
                        )

                        function_calls.append(
                            {
                                "name": name,
                                "arguments": arguments,
                                "id": response.get("id"),
                                "type": "openai",
                            }
                        )
                    except json.JSONDecodeError as e:
                        self._log_if_verbose(
                            "error",
                            f"Failed to parse arguments for {name}: {e}",
                        )

            # Check for choices[].message.tool_calls format
            choices = response.get("choices", [])
            for choice in choices:
                message = choice.get("message", {})
                tool_calls = message.get("tool_calls", [])

                for tool_call in tool_calls:
                    if tool_call.get("type") == "function":
                        function_info = tool_call.get("function", {})
                        name = function_info.get("name")
                        arguments_str = function_info.get(
                            "arguments", "{}"
                        )

                        if name:
                            try:
                                # Parse arguments JSON string
                                arguments = (
                                    json.loads(arguments_str)
                                    if isinstance(arguments_str, str)
                                    else arguments_str
                                )

                                function_calls.append(
                                    {
                                        "name": name,
                                        "arguments": arguments,
                                        "id": tool_call.get("id"),
                                        "type": "openai",
                                    }
                                )
                            except json.JSONDecodeError as e:
                                self._log_if_verbose(
                                    "error",
                                    f"Failed to parse arguments for {name}: {e}",
                                )

            # Also check for direct tool_calls in response root (array of function calls)
            if "tool_calls" in response:
                tool_calls = response["tool_calls"]
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if tool_call.get("type") == "function":
                            function_info = tool_call.get(
                                "function", {}
                            )
                            name = function_info.get("name")
                            arguments_str = function_info.get(
                                "arguments", "{}"
                            )

                            if name:
                                try:
                                    arguments = (
                                        json.loads(arguments_str)
                                        if isinstance(
                                            arguments_str, str
                                        )
                                        else arguments_str
                                    )

                                    function_calls.append(
                                        {
                                            "name": name,
                                            "arguments": arguments,
                                            "id": tool_call.get("id"),
                                            "type": "openai",
                                        }
                                    )
                                except json.JSONDecodeError as e:
                                    self._log_if_verbose(
                                        "error",
                                        f"Failed to parse arguments for {name}: {e}",
                                    )

        except Exception as e:
            self._log_if_verbose(
                "debug",
                f"Failed to extract OpenAI function calls: {e}",
            )

        return function_calls

    def _extract_anthropic_function_calls(
        self, response: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract function calls from Anthropic API response format."""
        function_calls = []

        try:
            # Check for content[].type == "tool_use" format
            content = response.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "tool_use"
                    ):
                        name = item.get("name")
                        input_data = item.get("input", {})

                        if name:
                            function_calls.append(
                                {
                                    "name": name,
                                    "arguments": input_data,
                                    "id": item.get("id"),
                                    "type": "anthropic",
                                }
                            )

            # Also check for direct tool_use format
            if response.get("type") == "tool_use":
                name = response.get("name")
                input_data = response.get("input", {})

                if name:
                    function_calls.append(
                        {
                            "name": name,
                            "arguments": input_data,
                            "id": response.get("id"),
                            "type": "anthropic",
                        }
                    )

            # Check for tool_calls array with Anthropic format (BaseModel converted)
            if "tool_calls" in response:
                tool_calls = response["tool_calls"]
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        # Handle BaseModel objects that have been converted to dict
                        if isinstance(tool_call, dict):
                            # Check for Anthropic-style function call
                            if (
                                tool_call.get("type") == "tool_use"
                                or "input" in tool_call
                            ):
                                name = tool_call.get("name")
                                input_data = tool_call.get(
                                    "input", {}
                                )

                                if name:
                                    function_calls.append(
                                        {
                                            "name": name,
                                            "arguments": input_data,
                                            "id": tool_call.get("id"),
                                            "type": "anthropic",
                                        }
                                    )
                            # Also check if it has function.name pattern but with input
                            elif "function" in tool_call:
                                function_info = tool_call.get(
                                    "function", {}
                                )
                                name = function_info.get("name")
                                # For Anthropic, prioritize 'input' over 'arguments'
                                input_data = function_info.get(
                                    "input"
                                ) or function_info.get(
                                    "arguments", {}
                                )

                                if name:
                                    function_calls.append(
                                        {
                                            "name": name,
                                            "arguments": input_data,
                                            "id": tool_call.get("id"),
                                            "type": "anthropic",
                                        }
                                    )

        except Exception as e:
            self._log_if_verbose(
                "debug",
                f"Failed to extract Anthropic function calls: {e}",
            )

        return function_calls

    def _extract_generic_function_calls(
        self, response: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract function calls from generic formats."""
        function_calls = []

        try:
            # Check if response itself is a function call
            if "name" in response and (
                "arguments" in response or "parameters" in response
            ):
                name = response.get("name")
                arguments = response.get("arguments") or response.get(
                    "parameters", {}
                )

                if name:
                    function_calls.append(
                        {
                            "name": name,
                            "arguments": arguments,
                            "id": response.get("id"),
                            "type": "generic",
                        }
                    )

            # Check for function_calls list
            if "function_calls" in response:
                for call in response["function_calls"]:
                    if isinstance(call, dict) and "name" in call:
                        name = call.get("name")
                        arguments = call.get("arguments") or call.get(
                            "parameters", {}
                        )

                        if name:
                            function_calls.append(
                                {
                                    "name": name,
                                    "arguments": arguments,
                                    "id": call.get("id"),
                                    "type": "generic",
                                }
                            )

        except Exception as e:
            self._log_if_verbose(
                "debug",
                f"Failed to extract generic function calls: {e}",
            )

        return function_calls

    def _execute_function_calls_sequential(
        self, function_calls: List[Dict[str, Any]]
    ) -> List[Any]:
        """Execute function calls sequentially."""
        results = []

        for i, call in enumerate(function_calls):
            try:
                self._log_if_verbose(
                    "info",
                    f"Executing function {call['name']} ({i+1}/{len(function_calls)})",
                )
                result = self._execute_single_function_call(call)
                results.append(result)
                self._log_if_verbose(
                    "info", f"Successfully executed {call['name']}"
                )
            except Exception as e:
                self._log_if_verbose(
                    "error", f"Failed to execute {call['name']}: {e}"
                )
                raise ToolExecutionError(
                    f"Failed to execute function {call['name']}: {e}"
                ) from e

        return results

    def _execute_function_calls_parallel(
        self, function_calls: List[Dict[str, Any]], max_workers: int
    ) -> List[Any]:
        """Execute function calls in parallel using concurrent.futures ThreadPoolExecutor."""
        self._log_if_verbose(
            "info",
            f"Executing {len(function_calls)} function calls in parallel with {max_workers} workers",
        )

        results = [None] * len(
            function_calls
        )  # Pre-allocate results list to maintain order

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all function calls to the executor
            future_to_index = {}
            for i, call in enumerate(function_calls):
                future = executor.submit(
                    self._execute_single_function_call, call
                )
                future_to_index[future] = i

            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                call = function_calls[index]

                try:
                    result = future.result()
                    results[index] = result
                    self._log_if_verbose(
                        "info",
                        f"Successfully executed {call['name']} (index {index})",
                    )
                except Exception as e:
                    self._log_if_verbose(
                        "error",
                        f"Failed to execute {call['name']} (index {index}): {e}",
                    )
                    raise ToolExecutionError(
                        f"Failed to execute function {call['name']}: {e}"
                    ) from e

        return results

    def _execute_single_function_call(
        self, call: Union[Dict[str, Any], BaseModel]
    ) -> Any:
        """Execute a single function call."""
        if isinstance(call, BaseModel):
            call = call.model_dump()

        name = call.get("name")
        arguments = call.get("arguments", {})

        if not name:
            raise ToolValidationError("Function call missing name")

        # Find the function
        if self.function_map and name in self.function_map:
            func = self.function_map[name]
        elif self.tools:
            func = self.find_function_name(name)
            if func is None:
                raise ToolNotFoundError(
                    f"Function {name} not found in tools"
                )
        else:
            raise ToolNotFoundError(f"Function {name} not found")

        # Execute the function
        try:
            if isinstance(arguments, dict):
                result = func(**arguments)
            else:
                result = func(arguments)
            return result
        except Exception as e:
            raise ToolExecutionError(
                f"Error executing function {name}: {e}"
            ) from e

    def detect_api_response_format(
        self, response: Union[Dict[str, Any], str, BaseModel]
    ) -> str:
        """
        Detect the format of an API response.

        Args:
            response: API response to analyze (can be BaseModel, dict, or string)

        Returns:
            str: Detected format ("openai", "anthropic", "generic", "unknown")

        Examples:
            >>> tool = BaseTool()
            >>> format_type = tool.detect_api_response_format(openai_response)
            >>> print(format_type)  # "openai"
        """
        # Handle BaseModel objects
        if isinstance(response, BaseModel):
            self._log_if_verbose(
                "debug",
                "Converting BaseModel response for format detection",
            )
            response = response.model_dump()

        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                return "unknown"

        if not isinstance(response, dict):
            return "unknown"

        # Check for single OpenAI function call object
        if (
            response.get("type") == "function"
            and "function" in response
        ):
            return "openai"

        # Check for OpenAI format with choices
        if "choices" in response:
            choices = response["choices"]
            if isinstance(choices, list) and len(choices) > 0:
                message = choices[0].get("message", {})
                if "tool_calls" in message:
                    return "openai"

        # Check for direct tool_calls array
        if "tool_calls" in response:
            return "openai"

        # Check for Anthropic format
        if "content" in response:
            content = response["content"]
            if isinstance(content, list):
                for item in content:
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "tool_use"
                    ):
                        return "anthropic"

        if response.get("type") == "tool_use":
            return "anthropic"

        # Check for generic format
        if "name" in response and (
            "arguments" in response
            or "parameters" in response
            or "input" in response
        ):
            return "generic"

        if "function_calls" in response:
            return "generic"

        return "unknown"

    def _extract_function_calls_from_tool_call_objects(
        self, tool_calls: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract function calls from a list of tool call objects (e.g., OpenAI ChatCompletionMessageToolCall or Anthropic BaseModels).

        Args:
            tool_calls: List of tool call objects (can include BaseModel objects)

        Returns:
            List[Dict[str, Any]]: List of standardized function call dictionaries
        """
        function_calls = []

        try:
            for tool_call in tool_calls:
                # Handle BaseModel objects (common with Anthropic responses)
                if isinstance(tool_call, BaseModel):
                    self._log_if_verbose(
                        "debug",
                        "Converting BaseModel tool call to dictionary",
                    )
                    tool_call_dict = tool_call.model_dump()

                    # Process the converted dictionary
                    extracted_calls = (
                        self._extract_function_calls_from_response(
                            tool_call_dict
                        )
                    )
                    function_calls.extend(extracted_calls)

                    # Also try direct extraction in case it's a simple function call BaseModel
                    if self._is_direct_function_call(tool_call_dict):
                        function_calls.extend(
                            self._extract_direct_function_call(
                                tool_call_dict
                            )
                        )

                # Handle OpenAI ChatCompletionMessageToolCall objects
                elif hasattr(tool_call, "function") and hasattr(
                    tool_call, "type"
                ):
                    if tool_call.type == "function":
                        function_info = tool_call.function
                        name = getattr(function_info, "name", None)
                        arguments_str = getattr(
                            function_info, "arguments", "{}"
                        )

                        if name:
                            try:
                                # Parse arguments JSON string
                                arguments = (
                                    json.loads(arguments_str)
                                    if isinstance(arguments_str, str)
                                    else arguments_str
                                )

                                function_calls.append(
                                    {
                                        "name": name,
                                        "arguments": arguments,
                                        "id": getattr(
                                            tool_call, "id", None
                                        ),
                                        "type": "openai",
                                    }
                                )
                            except json.JSONDecodeError as e:
                                self._log_if_verbose(
                                    "error",
                                    f"Failed to parse arguments for {name}: {e}",
                                )

                # Handle dictionary representations of tool calls
                elif isinstance(tool_call, dict):
                    if (
                        tool_call.get("type") == "function"
                        and "function" in tool_call
                    ):
                        function_info = tool_call["function"]
                        name = function_info.get("name")
                        arguments_str = function_info.get(
                            "arguments", "{}"
                        )

                        if name:
                            try:
                                arguments = (
                                    json.loads(arguments_str)
                                    if isinstance(arguments_str, str)
                                    else arguments_str
                                )

                                function_calls.append(
                                    {
                                        "name": name,
                                        "arguments": arguments,
                                        "id": tool_call.get("id"),
                                        "type": "openai",
                                    }
                                )
                            except json.JSONDecodeError as e:
                                self._log_if_verbose(
                                    "error",
                                    f"Failed to parse arguments for {name}: {e}",
                                )

                    # Also try other dictionary extraction methods
                    else:
                        extracted_calls = self._extract_function_calls_from_response(
                            tool_call
                        )
                        function_calls.extend(extracted_calls)

        except Exception as e:
            self._log_if_verbose(
                "error",
                f"Failed to extract function calls from tool call objects: {e}",
            )

        return function_calls

    def _format_results_as_strings(
        self, results: List[Any], function_calls: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Format function execution results as formatted strings.

        Args:
            results: List of function execution results
            function_calls: List of function call information

        Returns:
            List[str]: List of formatted result strings
        """
        formatted_results = []

        for i, (result, call) in enumerate(
            zip(results, function_calls)
        ):
            function_name = call.get("name", f"function_{i}")

            try:
                if isinstance(result, str):
                    formatted_result = f"Function '{function_name}' result:\n{result}"
                elif isinstance(result, dict):
                    formatted_result = f"Function '{function_name}' result:\n{json.dumps(result, indent=2, ensure_ascii=False)}"
                elif isinstance(result, (list, tuple)):
                    formatted_result = f"Function '{function_name}' result:\n{json.dumps(list(result), indent=2, ensure_ascii=False)}"
                else:
                    formatted_result = f"Function '{function_name}' result:\n{str(result)}"

                formatted_results.append(formatted_result)

            except Exception as e:
                self._log_if_verbose(
                    "error",
                    f"Failed to format result for {function_name}: {e}",
                )
                formatted_results.append(
                    f"Function '{function_name}' result: [Error formatting result: {str(e)}]"
                )

        return formatted_results

    def _is_direct_function_call(self, data: Dict[str, Any]) -> bool:
        """
        Check if a dictionary represents a direct function call.

        Args:
            data: Dictionary to check

        Returns:
            bool: True if it's a direct function call
        """
        return (
            isinstance(data, dict)
            and "name" in data
            and (
                "arguments" in data
                or "parameters" in data
                or "input" in data
            )
        )

    def _extract_direct_function_call(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract a direct function call from a dictionary.

        Args:
            data: Dictionary containing function call data

        Returns:
            List[Dict[str, Any]]: List containing the extracted function call
        """
        function_calls = []

        name = data.get("name")
        if name:
            # Try different argument key names
            arguments = (
                data.get("arguments")
                or data.get("parameters")
                or data.get("input")
                or {}
            )

            function_calls.append(
                {
                    "name": name,
                    "arguments": arguments,
                    "id": data.get("id"),
                    "type": "direct",
                }
            )

        return function_calls
