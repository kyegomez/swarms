from typing import Any
import inspect
from functools import partial
import logging


class NameResolver:
    """Utility class for resolving names of various objects"""

    @staticmethod
    def get_name(obj: Any, default: str = "unnamed_callable") -> str:
        """
        Get the name of any object with multiple fallback strategies.

        Args:
            obj: The object to get the name from
            default: Default name if all strategies fail

        Returns:
            str: The resolved name
        """
        strategies = [
            # Try getting __name__ attribute
            lambda x: getattr(x, "__name__", None),
            # Try getting class name
            lambda x: (
                x.__class__.__name__
                if hasattr(x, "__class__")
                else None
            ),
            # Try getting function name if it's a partial
            lambda x: (
                x.func.__name__ if isinstance(x, partial) else None
            ),
            # Try getting the name from the class's type
            lambda x: type(x).__name__,
            # Try getting qualname
            lambda x: getattr(x, "__qualname__", None),
            # Try getting the module and class name
            lambda x: (
                f"{x.__module__}.{x.__class__.__name__}"
                if hasattr(x, "__module__")
                else None
            ),
            # For async functions
            lambda x: (
                x.__name__ if inspect.iscoroutinefunction(x) else None
            ),
            # For classes with custom __str__
            lambda x: (
                str(x)
                if hasattr(x, "__str__")
                and x.__str__ != object.__str__
                else None
            ),
            # For wrapped functions
            lambda x: (
                getattr(x, "__wrapped__", None).__name__
                if hasattr(x, "__wrapped__")
                else None
            ),
        ]

        # Try each strategy
        for strategy in strategies:
            try:
                name = strategy(obj)
                if name and isinstance(name, str):
                    return name.replace(" ", "_").replace("-", "_")
            except Exception:
                continue

        # Return default if all strategies fail
        return default

    @staticmethod
    def get_callable_details(obj: Any) -> dict:
        """
        Get detailed information about a callable object.

        Returns:
            dict: Dictionary containing:
                - name: The resolved name
                - type: The type of callable
                - signature: The signature if available
                - module: The module name if available
                - doc: The docstring if available
        """
        details = {
            "name": NameResolver.get_name(obj),
            "type": "unknown",
            "signature": None,
            "module": getattr(obj, "__module__", "unknown"),
            "doc": inspect.getdoc(obj)
            or "No documentation available",
        }

        # Determine the type
        if inspect.isclass(obj):
            details["type"] = "class"
        elif inspect.iscoroutinefunction(obj):
            details["type"] = "async_function"
        elif inspect.isfunction(obj):
            details["type"] = "function"
        elif isinstance(obj, partial):
            details["type"] = "partial"
        elif callable(obj):
            details["type"] = "callable"

        # Try to get signature
        try:
            details["signature"] = str(inspect.signature(obj))
        except (ValueError, TypeError):
            details["signature"] = "Unknown signature"

        return details

    @classmethod
    def get_safe_name(cls, obj: Any, max_retries: int = 3) -> str:
        """
        Safely get a name with retries and validation.

        Args:
            obj: Object to get name from
            max_retries: Maximum number of retry attempts

        Returns:
            str: A valid name string
        """
        retries = 0
        last_error = None

        while retries < max_retries:
            try:
                name = cls.get_name(obj)

                # Validate and clean the name
                if name:
                    # Remove invalid characters
                    clean_name = "".join(
                        c
                        for c in name
                        if c.isalnum() or c in ["_", "."]
                    )

                    # Ensure it starts with a letter or underscore
                    if (
                        not clean_name[0].isalpha()
                        and clean_name[0] != "_"
                    ):
                        clean_name = f"_{clean_name}"

                    return clean_name

            except Exception as e:
                last_error = e
                retries += 1

        # If all retries failed, generate a unique fallback name
        import uuid

        fallback = f"callable_{uuid.uuid4().hex[:8]}"
        logging.warning(
            f"Failed to get name after {max_retries} retries. Using fallback: {fallback}. "
            f"Last error: {str(last_error)}"
        )
        return fallback


# # Example usage
# if __name__ == "__main__":
#     def test_resolver():
#         # Test cases
#         class TestClass:
#             def method(self):
#                 pass

#         async def async_func():
#             pass

#         test_cases = [
#             TestClass,                    # Class
#             TestClass(),                  # Instance
#             async_func,                   # Async function
#             lambda x: x,                  # Lambda
#             partial(print, end=""),       # Partial
#             TestClass.method,             # Method
#             print,                        # Built-in function
#             str,                          # Built-in class
#         ]

#         resolver = NameResolver()

#         print("\nName Resolution Results:")
#         print("-" * 50)
#         for obj in test_cases:
#             details = resolver.get_callable_details(obj)
#             safe_name = resolver.get_safe_name(obj)
#             print(f"\nObject: {obj}")
#             print(f"Safe Name: {safe_name}")
#             print(f"Details: {details}")

#     test_resolver()
