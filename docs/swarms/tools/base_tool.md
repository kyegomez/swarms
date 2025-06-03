# BaseTool Class Documentation

## Overview

The `BaseTool` class is a comprehensive tool management system for function calling, schema conversion, and execution. It provides a unified interface for converting Python functions to OpenAI function calling schemas, managing Pydantic models, executing tools with proper error handling, and supporting multiple AI provider formats (OpenAI, Anthropic, etc.).

**Key Features:**

- Convert Python functions to OpenAI function calling schemas

- Manage Pydantic models and their schemas  

- Execute tools with proper error handling and validation

- Support for parallel and sequential function execution

- Schema validation for multiple AI providers

- Automatic tool execution from API responses

- Caching for improved performance

## Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | `Optional[bool]` | `None` | Enable detailed logging output |
| `base_models` | `Optional[List[type[BaseModel]]]` | `None` | List of Pydantic models to manage |
| `autocheck` | `Optional[bool]` | `None` | Enable automatic validation checks |
| `auto_execute_tool` | `Optional[bool]` | `None` | Enable automatic tool execution |
| `tools` | `Optional[List[Callable[..., Any]]]` | `None` | List of callable functions to manage |
| `tool_system_prompt` | `Optional[str]` | `None` | System prompt for tool operations |
| `function_map` | `Optional[Dict[str, Callable]]` | `None` | Mapping of function names to callables |
| `list_of_dicts` | `Optional[List[Dict[str, Any]]]` | `None` | List of dictionary representations |

## Methods Overview

| Method | Description |
|--------|-------------|
| `func_to_dict` | Convert a callable function to OpenAI function calling schema |
| `load_params_from_func_for_pybasemodel` | Load function parameters for Pydantic BaseModel integration |
| `base_model_to_dict` | Convert Pydantic BaseModel to OpenAI schema dictionary |
| `multi_base_models_to_dict` | Convert multiple Pydantic BaseModels to OpenAI schema |
| `dict_to_openai_schema_str` | Convert dictionary to OpenAI schema string |
| `multi_dict_to_openai_schema_str` | Convert multiple dictionaries to OpenAI schema string |
| `get_docs_from_callable` | Extract documentation from callable items |
| `execute_tool` | Execute a tool based on response string |
| `detect_tool_input_type` | Detect the type of tool input |
| `dynamic_run` | Execute dynamic run with automatic type detection |
| `execute_tool_by_name` | Search for and execute tool by name |
| `execute_tool_from_text` | Execute tool from JSON-formatted string |
| `check_str_for_functions_valid` | Check if output is valid JSON with matching function |
| `convert_funcs_into_tools` | Convert all functions in tools list to OpenAI format |
| `convert_tool_into_openai_schema` | Convert tools into OpenAI function calling schema |
| `check_func_if_have_docs` | Check if function has proper documentation |
| `check_func_if_have_type_hints` | Check if function has proper type hints |
| `find_function_name` | Find function by name in tools list |
| `function_to_dict` | Convert function to dictionary representation |
| `multiple_functions_to_dict` | Convert multiple functions to dictionary representations |
| `execute_function_with_dict` | Execute function using dictionary of parameters |
| `execute_multiple_functions_with_dict` | Execute multiple functions with parameter dictionaries |
| `validate_function_schema` | Validate function schema for different AI providers |
| `get_schema_provider_format` | Get detected provider format of schema |
| `convert_schema_between_providers` | Convert schema between provider formats |
| `execute_function_calls_from_api_response` | Execute function calls from API responses |
| `detect_api_response_format` | Detect the format of API response |

---

## Detailed Method Documentation

### `func_to_dict`

**Description:** Convert a callable function to OpenAI function calling schema dictionary.

**Arguments:**
- `function` (Callable[..., Any], optional): The function to convert

**Returns:** `Dict[str, Any]` - OpenAI function calling schema dictionary

**Example:**
```python
from swarms.tools.base_tool import BaseTool

def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

# Create BaseTool instance
tool = BaseTool(verbose=True)

# Convert function to OpenAI schema
schema = tool.func_to_dict(add_numbers)
print(schema)
# Output: {'type': 'function', 'function': {'name': 'add_numbers', 'description': 'Add two numbers together.', 'parameters': {...}}}
```

### `load_params_from_func_for_pybasemodel`

**Description:** Load and process function parameters for Pydantic BaseModel integration.

**Arguments:**

- `func` (Callable[..., Any]): The function to process

- `*args`: Additional positional arguments

- `**kwargs`: Additional keyword arguments

**Returns:** `Callable[..., Any]` - Processed function with loaded parameters

**Example:**
```python
from swarms.tools.base_tool import BaseTool

def calculate_area(length: float, width: float) -> float:
    """Calculate area of a rectangle."""
    return length * width

tool = BaseTool()
processed_func = tool.load_params_from_func_for_pybasemodel(calculate_area)
```

### `base_model_to_dict`

**Description:** Convert a Pydantic BaseModel to OpenAI function calling schema dictionary.

**Arguments:**

- `pydantic_type` (type[BaseModel]): The Pydantic model class to convert

- `*args`: Additional positional arguments

- `**kwargs`: Additional keyword arguments

**Returns:** `dict[str, Any]` - OpenAI function calling schema dictionary

**Example:**
```python
from pydantic import BaseModel
from swarms.tools.base_tool import BaseTool

class UserInfo(BaseModel):
    name: str
    age: int
    email: str

tool = BaseTool()
schema = tool.base_model_to_dict(UserInfo)
print(schema)
```

### `multi_base_models_to_dict`

**Description:** Convert multiple Pydantic BaseModels to OpenAI function calling schema.

**Arguments:**
- `base_models` (List[BaseModel]): List of Pydantic models to convert

**Returns:** `dict[str, Any]` - Combined OpenAI function calling schema

**Example:**
```python
from pydantic import BaseModel
from swarms.tools.base_tool import BaseTool

class User(BaseModel):
    name: str
    age: int

class Product(BaseModel):
    name: str
    price: float

tool = BaseTool()
schemas = tool.multi_base_models_to_dict([User, Product])
print(schemas)
```

### `dict_to_openai_schema_str`

**Description:** Convert a dictionary to OpenAI function calling schema string.

**Arguments:**

- `dict` (dict[str, Any]): Dictionary to convert

**Returns:** `str` - OpenAI schema string representation

**Example:**
```python
from swarms.tools.base_tool import BaseTool

my_dict = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
    }
}

tool = BaseTool()
schema_str = tool.dict_to_openai_schema_str(my_dict)
print(schema_str)
```

### `multi_dict_to_openai_schema_str`

**Description:** Convert multiple dictionaries to OpenAI function calling schema string.

**Arguments:**

- `dicts` (list[dict[str, Any]]): List of dictionaries to convert

**Returns:** `str` - Combined OpenAI schema string representation

**Example:**
```python
from swarms.tools.base_tool import BaseTool

dict1 = {"type": "function", "function": {"name": "func1", "description": "Function 1"}}
dict2 = {"type": "function", "function": {"name": "func2", "description": "Function 2"}}

tool = BaseTool()
schema_str = tool.multi_dict_to_openai_schema_str([dict1, dict2])
print(schema_str)
```

### `get_docs_from_callable`

**Description:** Extract documentation from a callable item.

**Arguments:**

- `item`: The callable item to extract documentation from

**Returns:** Processed documentation

**Example:**
```python
from swarms.tools.base_tool import BaseTool

def example_function():
    """This is an example function with documentation."""
    pass

tool = BaseTool()
docs = tool.get_docs_from_callable(example_function)
print(docs)
```

### `execute_tool`

**Description:** Execute a tool based on a response string.

**Arguments:**
- `response` (str): JSON response string containing tool execution details

- `*args`: Additional positional arguments

- `**kwargs`: Additional keyword arguments

**Returns:** `Callable` - Result of the tool execution

**Example:**
```python
from swarms.tools.base_tool import BaseTool

def greet(name: str) -> str:
    """Greet a person by name."""
    return f"Hello, {name}!"

tool = BaseTool(tools=[greet])
response = '{"name": "greet", "parameters": {"name": "Alice"}}'
result = tool.execute_tool(response)
print(result)  # Output: "Hello, Alice!"
```

### `detect_tool_input_type`

**Description:** Detect the type of tool input for appropriate processing.

**Arguments:**

- `input` (ToolType): The input to analyze

**Returns:** `str` - Type of the input ("Pydantic", "Dictionary", "Function", or "Unknown")

**Example:**
```python
from swarms.tools.base_tool import BaseTool
from pydantic import BaseModel

class MyModel(BaseModel):
    value: int

def my_function():
    pass

tool = BaseTool()
print(tool.detect_tool_input_type(MyModel))  # "Pydantic"
print(tool.detect_tool_input_type(my_function))  # "Function"
print(tool.detect_tool_input_type({"key": "value"}))  # "Dictionary"
```

### `dynamic_run`

**Description:** Execute a dynamic run based on the input type with automatic type detection.

**Arguments:**
- `input` (Any): The input to be processed (Pydantic model, dict, or function)

**Returns:** `str` - The result of the dynamic run (schema string or execution result)

**Example:**
```python
from swarms.tools.base_tool import BaseTool

def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y

tool = BaseTool(auto_execute_tool=False)
result = tool.dynamic_run(multiply)
print(result)  # Returns OpenAI schema string
```

### `execute_tool_by_name`

**Description:** Search for a tool by name and execute it with the provided response.

**Arguments:**
- `tool_name` (str): The name of the tool to execute

- `response` (str): JSON response string containing execution parameters

**Returns:** `Any` - The result of executing the tool

**Example:**
```python
from swarms.tools.base_tool import BaseTool

def calculate_sum(a: int, b: int) -> int:
    """Calculate sum of two numbers."""
    return a + b

tool = BaseTool(function_map={"calculate_sum": calculate_sum})
result = tool.execute_tool_by_name("calculate_sum", '{"a": 5, "b": 3}')
print(result)  # Output: 8
```

### `execute_tool_from_text`

**Description:** Convert a JSON-formatted string into a tool dictionary and execute the tool.

**Arguments:**
- `text` (str): A JSON-formatted string representing a tool call with 'name' and 'parameters' keys

**Returns:** `Any` - The result of executing the tool

**Example:**
```python
from swarms.tools.base_tool import BaseTool

def divide(x: float, y: float) -> float:
    """Divide x by y."""
    return x / y

tool = BaseTool(function_map={"divide": divide})
text = '{"name": "divide", "parameters": {"x": 10, "y": 2}}'
result = tool.execute_tool_from_text(text)
print(result)  # Output: 5.0
```

### `check_str_for_functions_valid`

**Description:** Check if the output is a valid JSON string with a function name that matches the function map.

**Arguments:**
- `output` (str): The output string to validate

**Returns:** `bool` - True if the output is valid and the function name matches, False otherwise

**Example:**
```python
from swarms.tools.base_tool import BaseTool

def test_func():
    pass

tool = BaseTool(function_map={"test_func": test_func})
valid_output = '{"type": "function", "function": {"name": "test_func"}}'
is_valid = tool.check_str_for_functions_valid(valid_output)
print(is_valid)  # Output: True
```

### `convert_funcs_into_tools`

**Description:** Convert all functions in the tools list into OpenAI function calling format.

**Arguments:** None

**Returns:** None (modifies internal state)

**Example:**
```python
from swarms.tools.base_tool import BaseTool

def func1(x: int) -> int:
    """Function 1."""
    return x * 2

def func2(y: str) -> str:
    """Function 2."""
    return y.upper()

tool = BaseTool(tools=[func1, func2])
tool.convert_funcs_into_tools()
print(tool.function_map)  # {'func1': <function func1>, 'func2': <function func2>}
```

### `convert_tool_into_openai_schema`

**Description:** Convert tools into OpenAI function calling schema format.

**Arguments:** None

**Returns:** `dict[str, Any]` - Combined OpenAI function calling schema

**Example:**
```python
from swarms.tools.base_tool import BaseTool

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b

tool = BaseTool(tools=[add, subtract])
schema = tool.convert_tool_into_openai_schema()
print(schema)
```

### `check_func_if_have_docs`

**Description:** Check if a function has proper documentation.

**Arguments:**

- `func` (callable): The function to check

**Returns:** `bool` - True if function has documentation

**Example:**
```python
from swarms.tools.base_tool import BaseTool

def documented_func():
    """This function has documentation."""
    pass

def undocumented_func():
    pass

tool = BaseTool()
print(tool.check_func_if_have_docs(documented_func))  # True
# tool.check_func_if_have_docs(undocumented_func)  # Raises ToolDocumentationError
```

### `check_func_if_have_type_hints`

**Description:** Check if a function has proper type hints.

**Arguments:**

- `func` (callable): The function to check

**Returns:** `bool` - True if function has type hints

**Example:**
```python
from swarms.tools.base_tool import BaseTool

def typed_func(x: int) -> str:
    """A typed function."""
    return str(x)

def untyped_func(x):
    """An untyped function."""
    return str(x)

tool = BaseTool()
print(tool.check_func_if_have_type_hints(typed_func))  # True
# tool.check_func_if_have_type_hints(untyped_func)  # Raises ToolTypeHintError
```

### `find_function_name`

**Description:** Find a function by name in the tools list.

**Arguments:**
- `func_name` (str): The name of the function to find

**Returns:** `Optional[callable]` - The function if found, None otherwise

**Example:**
```python
from swarms.tools.base_tool import BaseTool

def my_function():
    """My function."""
    pass

tool = BaseTool(tools=[my_function])
found_func = tool.find_function_name("my_function")
print(found_func)  # <function my_function at ...>
```

### `function_to_dict`

**Description:** Convert a function to dictionary representation.

**Arguments:**
- `func` (callable): The function to convert

**Returns:** `dict` - Dictionary representation of the function

**Example:**
```python
from swarms.tools.base_tool import BaseTool

def example_func(param: str) -> str:
    """Example function."""
    return param

tool = BaseTool()
func_dict = tool.function_to_dict(example_func)
print(func_dict)
```

### `multiple_functions_to_dict`

**Description:** Convert multiple functions to dictionary representations.

**Arguments:**

- `funcs` (list[callable]): List of functions to convert

**Returns:** `list[dict]` - List of dictionary representations

**Example:**
```python
from swarms.tools.base_tool import BaseTool

def func1(x: int) -> int:
    """Function 1."""
    return x

def func2(y: str) -> str:
    """Function 2."""
    return y

tool = BaseTool()
func_dicts = tool.multiple_functions_to_dict([func1, func2])
print(func_dicts)
```

### `execute_function_with_dict`

**Description:** Execute a function using a dictionary of parameters.

**Arguments:**

- `func_dict` (dict): Dictionary containing function parameters

- `func_name` (Optional[str]): Name of function to execute (if not in dict)

**Returns:** `Any` - Result of function execution

**Example:**
```python
from swarms.tools.base_tool import BaseTool

def power(base: int, exponent: int) -> int:
    """Calculate base to the power of exponent."""
    return base ** exponent

tool = BaseTool(tools=[power])
result = tool.execute_function_with_dict({"base": 2, "exponent": 3}, "power")
print(result)  # Output: 8
```

### `execute_multiple_functions_with_dict`

**Description:** Execute multiple functions using dictionaries of parameters.

**Arguments:**

- `func_dicts` (list[dict]): List of dictionaries containing function parameters

- `func_names` (Optional[list[str]]): Optional list of function names

**Returns:** `list[Any]` - List of results from function executions

**Example:**
```python
from swarms.tools.base_tool import BaseTool

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

tool = BaseTool(tools=[add, multiply])
results = tool.execute_multiple_functions_with_dict(
    [{"a": 1, "b": 2}, {"a": 3, "b": 4}], 
    ["add", "multiply"]
)
print(results)  # [3, 12]
```

### `validate_function_schema`

**Description:** Validate the schema of a function for different AI providers.

**Arguments:**

- `schema` (Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]): Function schema(s) to validate

- `provider` (str): Target provider format ("openai", "anthropic", "generic", "auto")

**Returns:** `bool` - True if schema(s) are valid, False otherwise

**Example:**
```python
from swarms.tools.base_tool import BaseTool

openai_schema = {
    "type": "function",
    "function": {
        "name": "add_numbers",
        "description": "Add two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"}
            },
            "required": ["a", "b"]
        }
    }
}

tool = BaseTool()
is_valid = tool.validate_function_schema(openai_schema, "openai")
print(is_valid)  # True
```

### `get_schema_provider_format`

**Description:** Get the detected provider format of a schema.

**Arguments:**

- `schema` (Dict[str, Any]): Function schema dictionary

**Returns:** `str` - Provider format ("openai", "anthropic", "generic", "unknown")

**Example:**
```python
from swarms.tools.base_tool import BaseTool

openai_schema = {
    "type": "function",
    "function": {"name": "test", "description": "Test function"}
}

tool = BaseTool()
provider = tool.get_schema_provider_format(openai_schema)
print(provider)  # "openai"
```

### `convert_schema_between_providers`

**Description:** Convert a function schema between different provider formats.

**Arguments:**

- `schema` (Dict[str, Any]): Source function schema

- `target_provider` (str): Target provider format ("openai", "anthropic", "generic")

**Returns:** `Dict[str, Any]` - Converted schema

**Example:**
```python
from swarms.tools.base_tool import BaseTool

openai_schema = {
    "type": "function",
    "function": {
        "name": "test_func",
        "description": "Test function",
        "parameters": {"type": "object", "properties": {}}
    }
}

tool = BaseTool()
anthropic_schema = tool.convert_schema_between_providers(openai_schema, "anthropic")
print(anthropic_schema)
# Output: {"name": "test_func", "description": "Test function", "input_schema": {...}}
```

### `execute_function_calls_from_api_response`

**Description:** Automatically detect and execute function calls from OpenAI or Anthropic API responses.

**Arguments:**

- `api_response` (Union[Dict[str, Any], str, List[Any]]): The API response containing function calls

- `sequential` (bool): If True, execute functions sequentially. If False, execute in parallel

- `max_workers` (int): Maximum number of worker threads for parallel execution

- `return_as_string` (bool): If True, return results as formatted strings

**Returns:** `Union[List[Any], List[str]]` - List of results from executed functions

**Example:**
```python
from swarms.tools.base_tool import BaseTool

def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 25°C"

# Simulated OpenAI API response
openai_response = {
    "choices": [{
        "message": {
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "New York"}'
                },
                "id": "call_123"
            }]
        }
    }]
}

tool = BaseTool(tools=[get_weather])
results = tool.execute_function_calls_from_api_response(openai_response)
print(results)  # ["Function 'get_weather' result:\nWeather in New York: Sunny, 25°C"]
```

### `detect_api_response_format`

**Description:** Detect the format of an API response.

**Arguments:**

- `response` (Union[Dict[str, Any], str, BaseModel]): API response to analyze

**Returns:** `str` - Detected format ("openai", "anthropic", "generic", "unknown")

**Example:**
```python
from swarms.tools.base_tool import BaseTool

openai_response = {
    "choices": [{"message": {"tool_calls": []}}]
}

anthropic_response = {
    "content": [{"type": "tool_use", "name": "test", "input": {}}]
}

tool = BaseTool()
print(tool.detect_api_response_format(openai_response))  # "openai"
print(tool.detect_api_response_format(anthropic_response))  # "anthropic"
```

---

## Exception Classes

The BaseTool class defines several custom exception classes for better error handling:

- `BaseToolError`: Base exception class for all BaseTool related errors

- `ToolValidationError`: Raised when tool validation fails

- `ToolExecutionError`: Raised when tool execution fails

- `ToolNotFoundError`: Raised when a requested tool is not found

- `FunctionSchemaError`: Raised when function schema conversion fails

- `ToolDocumentationError`: Raised when tool documentation is missing or invalid

- `ToolTypeHintError`: Raised when tool type hints are missing or invalid

## Usage Tips

1. **Always provide documentation and type hints** for your functions when using BaseTool
2. **Use verbose=True** during development for detailed logging
3. **Set up function_map** for efficient tool execution by name
4. **Validate schemas** before using them with different AI providers
5. **Use parallel execution** for better performance when executing multiple functions
6. **Handle exceptions** appropriately using the custom exception classes 