"""
Example usage of the log_function_execution decorator.

This example demonstrates how to use the decorator to automatically log
function executions including parameters, outputs, and execution metadata.
"""

from swarms.telemetry.log_executions import log_function_execution


# Example 1: Simple function with basic parameters
@log_function_execution(
    swarm_id="example-swarm-001",
    swarm_architecture="sequential",
    enabled_on=True,
)
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b


# Example 2: Function with complex parameters and return values
@log_function_execution(
    swarm_id="data-processing-swarm",
    swarm_architecture="parallel",
    enabled_on=True,
)
def process_data(
    data_list: list,
    threshold: float = 0.5,
    include_metadata: bool = True,
) -> dict:
    """Process a list of data with filtering and metadata generation."""
    filtered_data = [x for x in data_list if x > threshold]

    result = {
        "original_count": len(data_list),
        "filtered_count": len(filtered_data),
        "filtered_data": filtered_data,
        "threshold_used": threshold,
    }

    if include_metadata:
        result["metadata"] = {
            "processing_method": "threshold_filter",
            "success": True,
        }

    return result


# Example 3: Function that might raise an exception
@log_function_execution(
    swarm_id="validation-swarm",
    swarm_architecture="error_handling",
    enabled_on=True,
)
def validate_input(value: str, min_length: int = 5) -> bool:
    """Validate input string length."""
    if not isinstance(value, str):
        raise TypeError(f"Expected string, got {type(value)}")

    if len(value) < min_length:
        raise ValueError(
            f"String too short: {len(value)} < {min_length}"
        )

    return True


# Example 4: Decorator with logging disabled
@log_function_execution(
    swarm_id="silent-swarm",
    swarm_architecture="background",
    enabled_on=False,  # Logging disabled
)
def silent_function(x: int) -> int:
    """This function won't be logged."""
    return x * 2


if __name__ == "__main__":
    print("Testing log_function_execution decorator...")

    # Test successful executions
    print("\n1. Testing simple sum calculation:")
    result1 = calculate_sum(5, 3)
    print(f"Result: {result1}")

    print("\n2. Testing data processing:")
    sample_data = [0.2, 0.7, 1.2, 0.1, 0.9, 1.5]
    result2 = process_data(
        sample_data, threshold=0.5, include_metadata=True
    )
    print(f"Result: {result2}")

    print("\n3. Testing validation with valid input:")
    result3 = validate_input("hello world", min_length=5)
    print(f"Result: {result3}")

    print("\n4. Testing silent function (no logging):")
    result4 = silent_function(10)
    print(f"Result: {result4}")

    print(
        "\n5. Testing validation with invalid input (will raise exception):"
    )
    try:
        validate_input("hi", min_length=5)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\nAll function calls have been logged automatically!")
    print(
        "Check your telemetry logs to see the captured execution data."
    )
