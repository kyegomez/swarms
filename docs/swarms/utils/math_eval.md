# Math Evaluation Decorator Documentation

## Introduction
The Math Evaluation Decorator is a utility function that helps you compare the output of two functions, `func1` and `func2`, when given the same input. This decorator is particularly useful for validating whether a generated function produces the same results as a ground truth function. This documentation provides a detailed explanation of the Math Evaluation Decorator, its purpose, usage, and examples.

## Purpose
The Math Evaluation Decorator serves the following purposes:
1. To compare the output of two functions, `func1` and `func2`, when given the same input.
2. To log any errors that may occur during the evaluation.
3. To provide a warning if the outputs of `func1` and `func2` do not match.

## Decorator Definition
```python
def math_eval(func1, func2):
    """Math evaluation decorator.

    Args:
        func1 (_type_): The first function to be evaluated.
        func2 (_type_): The second function to be evaluated.

    Example:
    >>> @math_eval(ground_truth, generated_func)
    >>> def test_func(x):
    >>>     return x
    >>> result1, result2 = test_func(5)
    >>> print(f"Result from ground_truth: {result1}")
    >>> print(f"Result from generated_func: {result2}")

    """
```

### Parameters
| Parameter | Type   | Description                                      |
|-----------|--------|--------------------------------------------------|
| `func1`   | _type_ | The first function to be evaluated.             |
| `func2`   | _type_ | The second function to be evaluated.            |

## Usage
The Math Evaluation Decorator is used as a decorator for a test function that you want to evaluate. Here's how to use it:

1. Define the two functions, `func1` and `func2`, that you want to compare.

2. Create a test function and decorate it with `@math_eval(func1, func2)`.

3. In the test function, provide the input(s) to both `func1` and `func2`.

4. The decorator will compare the outputs of `func1` and `func2` when given the same input(s).

5. Any errors that occur during the evaluation will be logged.

6. If the outputs of `func1` and `func2` do not match, a warning will be generated.

## Examples

### Example 1: Comparing Two Simple Functions
```python
# Define the ground truth function
def ground_truth(x):
    return x * 2

# Define the generated function
def generated_func(x):
    return x - 10

# Create a test function and decorate it
@math_eval(ground_truth, generated_func)
def test_func(x):
    return x

# Evaluate the test function with an input
result1, result2 = test_func(5)

# Print the results
print(f"Result from ground_truth: {result1}")
print(f"Result from generated_func: {result2}")
```

In this example, the decorator compares the outputs of `ground_truth` and `generated_func` when given the input `5`. If the outputs do not match, a warning will be generated.

### Example 2: Handling Errors
If an error occurs in either `func1` or `func2`, the decorator will log the error and set the result to `None`. This ensures that the evaluation continues even if one of the functions encounters an issue.

## Additional Information and Tips

- The Math Evaluation Decorator is a powerful tool for comparing the outputs of functions, especially when validating machine learning models or generated code.

- Ensure that the functions `func1` and `func2` take the same input(s) to ensure a meaningful comparison.

- Regularly check the logs for any errors or warnings generated during the evaluation.

- If the decorator logs a warning about mismatched outputs, investigate and debug the functions accordingly.

## References and Resources

- For more information on Python decorators, refer to the [Python Decorators Documentation](https://docs.python.org/3/glossary.html#term-decorator).

- Explore advanced use cases of the Math Evaluation Decorator in your projects to ensure code correctness and reliability.

This comprehensive documentation explains the Math Evaluation Decorator, its purpose, usage, and examples. Use this decorator to compare the outputs of functions and validate code effectively.