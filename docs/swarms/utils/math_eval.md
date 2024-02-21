# math_eval


The `math_eval` function is a python decorator that wraps around a function to run two functions on the same inputs and compare their results. The decorator can be used for testing functions that are expected to have equivalent functionality, or in situations where two different methods are used to calculate or retrieve a value, and the results need to be compared.

The `math_eval` function in this case accepts two functions as parameters: `func1` and `func2`, and returns a decorator. This returned decorator, when applied to a function, enhances that function to execute both `func1` and `func2`, and compare the results.

This can be particularly useful in situations when you are implementing a new function and wants to compare its behavior and results with that of an existing one under the same set of input parameters. It also logs the results if they do not match which could be quite useful during the debug process.

## Usage Example

Let's say you have two functions: `ground_truth` and `generated_func`, that have similar functionalities or serve the same purpose. You are writing a new function called `test_func`, and you'd like to compare the results of `ground_truth` and `generated_func` when `test_func` is run. Here is how you would use the `math_eval` decorator:

```python
@math_eval(ground_truth, generated_func)
def test_func(x):
    return x


result1, result2 = test_func(5)
print(f"Result from ground_truth: {result1}")
print(f"Result from generated_func: {result2}")
```

## Parameters

| Parameter | Data Type | Description |
| ---- | ---- | ---- |
| func1 | Callable | The first function whose result you want to compare. |
| func2 | Callable | The second function whose result you want to compare. |

The data types for `func1` and `func2` cannot be specified as they can be any python function (or callable object). The decorator verifies that they are callable and exceptions are handled within the decorator function.

## Return Values

The `math_eval` function does not return a direct value, since it is a decorator. When applied to a function, it alters the behavior of the wrapped function to return two values:

1. `result1`: The result of running `func1` with the given input parameters.
2. `result2`: The result of running `func2` with the given input parameters.

These two return values are provided in that order as a tuple.

## Source Code

Here's how to implement the `math_eval` decorator:

```python
import functools
import logging


def math_eval(func1, func2):
    """Math evaluation decorator."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result1 = func1(*args, **kwargs)
            except Exception as e:
                logging.error(f"Error in func1: {e}")
                result1 = None

            try:
                result2 = func2(*args, **kwargs)
            except Exception as e:
                logging.error(f"Error in func2: {e}")
                result2 = None

            if result1 != result2:
                logging.warning(f"Outputs do not match: {result1} != {result2}")

            return result1, result2

        return wrapper

    return decorator
```
Please note that the code is logging exceptions to facilitate debugging, but the actual processing and handling of the exception would depend on how you want your application to respond to exceptions. Therefore, you may want to customize the error handling depending upon your application's requirements.
