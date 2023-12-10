# Import necessary modules and functions for testing
import functools
import subprocess
import sys
import traceback

import pytest

# Try importing phoenix and handle exceptions
try:
    import phoenix as px
except Exception as error:
    print(f"Error importing phoenix: {error}")
    print("Please install phoenix: pip install phoenix")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "arize-mlflow"]
    )

# Import the code to be tested
from swarms.utils.phoenix_handler import phoenix_trace_decorator


# Define a fixture for Phoenix session
@pytest.fixture(scope="function")
def phoenix_session():
    session = px.active_session() or px.launch_app()
    yield session
    session.stop()


# Define test cases for the phoenix_trace_decorator function
def test_phoenix_trace_decorator_documentation():
    """Test if phoenix_trace_decorator has a docstring."""
    assert phoenix_trace_decorator.__doc__ is not None


def test_phoenix_trace_decorator_functionality(
    capsys, phoenix_session
):
    """Test the functionality of phoenix_trace_decorator."""

    # Define a function to be decorated
    @phoenix_trace_decorator("This is a test function.")
    def test_function():
        print("Hello, Phoenix!")

    # Execute the decorated function
    test_function()

    # Capture the printed output
    captured = capsys.readouterr()
    assert captured.out == "Hello, Phoenix!\n"


def test_phoenix_trace_decorator_exception_handling(phoenix_session):
    """Test if phoenix_trace_decorator handles exceptions correctly."""

    # Define a function that raises an exception
    @phoenix_trace_decorator("This function raises an exception.")
    def exception_function():
        raise ValueError("An error occurred.")

    # Execute the decorated function
    with pytest.raises(ValueError):
        exception_function()

    # Check if the exception was traced by Phoenix
    traces = phoenix_session.get_traces()
    assert len(traces) == 1
    assert traces[0].get("error") is not None
    assert traces[0].get("error_info") is not None


# Define test cases for phoenix_trace_decorator
def test_phoenix_trace_decorator_docstring():
    """Test if phoenix_trace_decorator's inner function has a docstring."""

    @phoenix_trace_decorator("This is a test function.")
    def test_function():
        """Test function docstring."""
        pass

    assert test_function.__doc__ is not None


def test_phoenix_trace_decorator_functionality_with_params(
    capsys, phoenix_session
):
    """Test the functionality of phoenix_trace_decorator with parameters."""

    # Define a function with parameters to be decorated
    @phoenix_trace_decorator("This function takes parameters.")
    def param_function(a, b):
        result = a + b
        print(f"Result: {result}")

    # Execute the decorated function with parameters
    param_function(2, 3)

    # Capture the printed output
    captured = capsys.readouterr()
    assert captured.out == "Result: 5\n"


def test_phoenix_trace_decorator_nested_calls(
    capsys, phoenix_session
):
    """Test nested calls of phoenix_trace_decorator."""

    # Define a nested function with decorators
    @phoenix_trace_decorator("Outer function")
    def outer_function():
        print("Outer function")

        @phoenix_trace_decorator("Inner function")
        def inner_function():
            print("Inner function")

        inner_function()

    # Execute the decorated functions
    outer_function()

    # Capture the printed output
    captured = capsys.readouterr()
    assert "Outer function" in captured.out
    assert "Inner function" in captured.out


def test_phoenix_trace_decorator_nested_exception_handling(
    phoenix_session,
):
    """Test exception handling with nested phoenix_trace_decorators."""

    # Define a function with nested decorators and an exception
    @phoenix_trace_decorator("Outer function")
    def outer_function():
        @phoenix_trace_decorator("Inner function")
        def inner_function():
            raise ValueError("Inner error")

        inner_function()

    # Execute the decorated functions
    with pytest.raises(ValueError):
        outer_function()

    # Check if both exceptions were traced by Phoenix
    traces = phoenix_session.get_traces()
    assert len(traces) == 2
    assert "Outer function" in traces[0].get("error_info")
    assert "Inner function" in traces[1].get("error_info")
