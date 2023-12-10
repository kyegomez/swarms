# Phoenix Trace Decorator Documentation

## Introduction

Welcome to the documentation for the `phoenix_trace_decorator` module. This module provides a convenient decorator for tracing Python functions and capturing exceptions using Phoenix, a versatile tracing and monitoring tool. Phoenix allows you to gain insights into the execution of your code, capture errors, and monitor performance.

## Table of Contents

1. [Installation](#installation)
2. [Getting Started](#getting-started)
3. [Decorator Usage](#decorator-usage)
4. [Examples](#examples)
5. [Best Practices](#best-practices)
6. [References](#references)

## 1. Installation <a name="installation"></a>

Before using the `phoenix_trace_decorator`, you need to install the Swarms library. You can install Phoenix using pip:

```bash
pip install swarms
```

## 2. Getting Started <a name="getting-started"></a>

Phoenix is a powerful tracing and monitoring tool, and the `phoenix_trace_decorator` simplifies the process of tracing functions and capturing exceptions within your Python code. To begin, ensure that Phoenix is installed, and then import the `phoenix_trace_decorator` module into your Python script.

```python
from swarms import phoenix_trace_decorator
```

## 3. Decorator Usage <a name="decorator-usage"></a>

The `phoenix_trace_decorator` module provides a decorator, `phoenix_trace_decorator`, which can be applied to functions you want to trace. The decorator takes a single argument, a docstring that describes the purpose of the function being traced.

Here is the basic structure of using the decorator:

```python
@phoenix_trace_decorator("Description of the function")
def my_function(param1, param2):
    # Function implementation
    pass
```

## 4. Examples <a name="examples"></a>

Let's explore some practical examples of using the `phoenix_trace_decorator` in your code.

### Example 1: Basic Tracing

In this example, we'll trace a simple function and print a message.

```python
@phoenix_trace_decorator("Tracing a basic function")
def hello_world():
    print("Hello, World!")

# Call the decorated function
hello_world()
```

### Example 2: Tracing a Function with Parameters

You can use the decorator with functions that have parameters.

```python
@phoenix_trace_decorator("Tracing a function with parameters")
def add_numbers(a, b):
    result = a + b
    print(f"Result: {result}")

# Call the decorated function with parameters
add_numbers(2, 3)
```

### Example 3: Tracing Nested Calls

The decorator can also trace nested function calls.

```python
@phoenix_trace_decorator("Outer function")
def outer_function():
    print("Outer function")

    @phoenix_trace_decorator("Inner function")
    def inner_function():
        print("Inner function")

    inner_function()

# Call the decorated functions
outer_function()
```

### Example 4: Exception Handling

Phoenix can capture exceptions and provide detailed information about them.

```python
@phoenix_trace_decorator("Function with exception handling")
def divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError as e:
        raise ValueError("Division by zero") from e

# Call the decorated function with an exception
try:
    divide(5, 0)
except ValueError as e:
    print(f"Error: {e}")
```

## 5. Best Practices <a name="best-practices"></a>

When using the `phoenix_trace_decorator`, consider the following best practices:

- Use meaningful docstrings to describe the purpose of the traced functions.
- Keep your tracing focused on critical parts of your code.
- Make sure Phoenix is properly configured and running before using the decorator.

## 6. References <a name="references"></a>

For more information on Phoenix and advanced usage, please refer to the [Phoenix Documentation](https://phoenix-docs.readthedocs.io/en/latest/).

---

By following this documentation, you can effectively use the `phoenix_trace_decorator` to trace your Python functions, capture exceptions, and gain insights into the execution of your code. This tool is valuable for debugging, performance optimization, and monitoring the health of your applications.