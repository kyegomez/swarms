# print_class_parameters

# Module Function Name: print_class_parameters

The `print_class_parameters` function is a utility function developed to help developers and users alike in retrieving and printing the parameters of a class constructor in Python, either in standard output or returned as a dictionary if the `api_format` is set to `True`.

This utility function utilizes the `inspect` module to fetch the signature of the class constructor and fetches the parameters from the obtained signature. The parameter values and their respective types are then outputted.

This function allows developers to easily inspect and understand the class' constructor parameters without the need to individually go through the class structure. This eases the testing and debugging process for developers and users alike, aiding in generating more efficient and readable code.

__Function Definition:__

```python
def print_class_parameters(cls, api_format: bool = False):
```
__Parameters:__

| Parameter  | Type   | Description  | Default value |
|---|---|---|---|
| cls  | type  | The Python class to inspect.  | None |
| api_format  | bool  | Flag to determine if the output should be returned in dictionary format (if set to True) or printed out (if set to False) | False |

__Functionality and Usage:__

Inside the `print_class_parameters` function, it starts by getting the signature of the constructor of the inputted class by invoking `inspect.signature(cls.__init__)`. It then extracts the parameters from the signature and stores it in the `params` variable.

If the `api_format` argument is set to `True`, instead of printing the parameters and their types, it stores them inside a dictionary where each key-value pair is a parameter name and its type. It then returns this dictionary.

If `api_format` is set to `False` or not set at all (defaulting to False), the function iterates over the parameters and prints the parameter name and its type. "self" parameters are excluded from the output as they are inherent to all class methods in Python.

A possible exception that may occur during the execution of this function is during the invocation of the `inspect.signature()` function call. If the inputted class does not have an `__init__` method or any error occurs during the retrieval of the class constructor's signature, an exception will be triggered. In that case, an error message that includes the error details is printed out.

__Usage and Examples:__

Assuming the existence of a class:

```python
class Agent:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
```

One could use `print_class_parameters` in its typical usage:

```python
print_class_parameters(Agent)
```

Results in:

```
Parameter: x, Type: <class 'int'>
Parameter: y, Type: <class 'int'>
```

Or, with `api_format` set to `True`

```python
output = print_class_parameters(Agent, api_format=True)
print(output)
```

Results in:

```
{'x': "<class 'int'>", 'y': "<class 'int'>"}
```

__Note:__

The function `print_class_parameters` is not limited to custom classes. It can inspect built-in Python classes such as `list`, `dict`, and others. However, it is most useful when inspecting custom-defined classes that aren't inherently documented in Python or third-party libraries.

__Source Code__

```python
def print_class_parameters(cls, api_format: bool = False):
    """
    Print the parameters of a class constructor.

    Parameters:
    cls (type): The class to inspect.

    Example:
    >>> print_class_parameters(Agent)
    Parameter: x, Type: <class 'int'>
    Parameter: y, Type: <class 'int'>
    """
    try:
        # Get the parameters of the class constructor
        sig = inspect.signature(cls.__init__)
        params = sig.parameters

        if api_format:
            param_dict = {}
            for name, param in params.items():
              if name == "self":
                continue
              param_dict[name] = str(param.annotation)
            return param_dict

        # Print the parameters
        for name, param in params.items():
          if name == "self":
            continue
          print(f"Parameter: {name}, Type: {param.annotation}")

    except Exception as e:
        print(f"An error occurred while inspecting the class: {e}")
```
