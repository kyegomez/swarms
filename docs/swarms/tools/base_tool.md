# Documentation Outline for `BaseTool` Class

1. **Module Overview**
2. **Installation and Setup**
3. **Class Definition**
4. **Attributes and Methods**
5. **Functionality and Usage**
   - Basic Usage Examples
   - Advanced Use Cases
6. **Common Issues and Troubleshooting**
7. **References and Additional Resources**

## 1. Module Overview

The `BaseTool` class is a part of the `swarms` package and serves as a foundational class for creating and managing tools that can be executed with different inputs and configurations. It leverages Pydantic for input validation and includes extensive logging for easy debugging and monitoring.

## 2. Installation and Setup

To use the `BaseTool` class, ensure that you have the required dependencies installed:

```bash
pip install pydantic loguru
```

Include the necessary imports in your Python script:

```python
from swarms.tools.base_tool import BaseTool
```

## 3. Class Definition

`BaseTool` is designed using Pydantic's `BaseModel` to leverage type annotations and validations:

```python
from pydantic import BaseModel

class BaseTool(BaseModel):
    # Attributes and method definitions follow
```

## 4. Attributes and Methods

### Attributes

| Attribute           | Type                       | Description                                                  |
|---------------------|----------------------------|--------------------------------------------------------------|
| `verbose`           | `bool`                     | Enables verbose output, providing detailed logs.             |
| `functions`         | `List[Callable[..., Any]]` | Stores a list of functions that can be managed by the tool.  |
| `base_models`       | `List[type[BaseModel]]`    | List of Pydantic models associated with the tool.            |
| `autocheck`         | `bool`                     | Automatically checks conditions before execution (not implemented). |
| `auto_execute_tool` | `Optional[bool]`           | Automatically executes tools if set.                         |

### Key Methods

- `func_to_dict`: Converts a function to a dictionary format suitable for OpenAI function schema.
- `load_params_from_func_for_pybasemodel`: Loads parameters dynamically for Pydantic models based on the function signature.
- `execute_tool`: Executes a specified tool using a mapping of function names to callable functions.

## 5. Functionality and Usage

### Basic Usage Examples

#### Initialize BaseTool

```python
tool = BaseTool(verbose=True)
```

#### Convert a Function to Dictionary

```python
def sample_function(x, y):
    return x + y

schema = tool.func_to_dict(sample_function, name="AddFunction", description="Adds two numbers")
print(schema)
```

### Advanced Use Cases

#### Executing a Tool Dynamically

```python
# Define a sample tool
def add(x, y):
    return x + y

# Tool registration and execution
tool_dict = tool.func_to_dict(add, name="Add")
result = tool.execute_tool([tool_dict], {'Add': add}, 5, 3)
print("Result of add:", result)
```

#### Handling Multiple Models

```python
# Define multiple Pydantic models
class ModelOne(BaseModel):
    a: int

class ModelTwo(BaseModel):
    b: str

# Convert and manage multiple models
schemas = tool.multi_base_models_to_dict([ModelOne, ModelTwo])
print(schemas)
```

## 6. Common Issues and Troubleshooting

- **Type Errors**: Ensure that all parameters match the expected types as defined in the Pydantic models.
- **Execution Failures**: Check the function and tool configurations for compatibility and completeness.

## 7. References and Additional Resources

- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)
- [Loguru GitHub Repository](https://github.com/Delgan/loguru)
