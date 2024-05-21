
# Tool Decorator Documentation

## Module Overview

The `tool` decorator is designed to enhance functions by automatically generating an OpenAI function schema based on the function's signature and provided metadata. This schema can be outputted in different formats based on the decorator's arguments. The primary use of this decorator is to facilitate the integration of Python functions with external systems that require structured metadata, making it ideal for creating machine-readable descriptions of functions.

## Key Features

- **Automatic Schema Generation:** Generates a schema based on the function's signature.
- **Flexible Output Formats:** Supports returning the schema as a dictionary, string, or YAML (if integrated).
- **Logging Support:** Includes logging of function calls and errors, aiding in debugging and monitoring.

## Installation and Setup

Before using the `tool` decorator, ensure that the required libraries are installed and configured. Here’s a basic setup:

```bash
$ pip install -U swarms
```

## Decorator Definition

### Signature

```python
def tool(name: str = None, description: str = None, return_dict: bool = True, verbose: bool = True, return_string: bool = False, return_yaml: bool = False):
```

### Parameters

| Parameter        | Type    | Default | Description                                            |
|------------------|---------|---------|--------------------------------------------------------|
| `name`           | str     | None    | Name of the OpenAI function. Optional.                 |
| `description`    | str     | None    | Description of the OpenAI function. Optional.          |
| `return_dict`    | bool    | True    | Whether to return the schema as a dictionary.          |
| `verbose`        | bool    | True    | Enables verbose output.                                |
| `return_string`  | bool    | False   | Whether to return the schema as a string.              |
| `return_yaml`    | bool    | False   | Whether to return the schema in YAML format.           |

## Functionality and Usage

### Basic Usage

Here is an example of using the `tool` decorator to enhance a simple function:

```python
@tool(name="ExampleFunction", description="Demonstrates the use of the tool decorator")
def example_function(param1: int, param2: str):
    print(f"Received param1: {param1}, param2: {param2}")

example_function(123, "abc")
```

### Advanced Usage

#### Returning Schema as String

To get the schema as a string instead of a dictionary:

```python
@tool(name="StringSchemaFunction", description="Returns schema as string", return_dict=False, return_string=True)
def another_function():
    pass

print(another_function())  # Outputs the schema as a string
```

#### Handling Exceptions

Demonstrating error handling with the decorator:

```python
@tool(name="ErrorHandlingFunction", description="Handles errors gracefully")
def error_prone_function():
    raise ValueError("An example error")

try:
    error_prone_function()
except Exception as e:
    print(f"Caught an error: {e}")
```

## Additional Information and Tips

- **Logging:** The decorator logs all function calls and exceptions. Make sure to configure the `loguru` logger accordingly to capture these logs.
- **Assertion Errors:** The decorator performs type checks on the arguments, and if the types do not match, it will raise an assertion error.

## References

- For more on decorators: [Python Decorators Documentation](https://docs.python.org/3/glossary.html#term-decorator)
- Loguru library for logging: [Loguru Documentation](https://loguru.readthedocs.io/en/stable/)
