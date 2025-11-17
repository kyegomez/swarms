from swarms.tools.tool_parse_exec import parse_and_execute_json


def run_test(test_name, test_func):
    print(f"Running test: {test_name}")
    print("------------------------------------------------")
    try:
        test_func()
        print(f"✓ {test_name} passed")
        print("------------------------------------------------")
    except Exception as e:
        print(f"✗ {test_name} failed: {str(e)}")
        print("------------------------------------------------")


# Mock functions for testing
def mock_function_a(param1, param2):
    return param1 + param2


def mock_function_b(param1):
    if param1 < 0:
        raise ValueError("Negative value not allowed")
    return param1 * 2


# Test cases
def test_parse_and_execute_json_success():
    functions = [mock_function_a, mock_function_b]
    json_string = '{"functions": [{"name": "mock_function_a", "parameters": {"param1": 1, "param2": 2}}, {"name": "mock_function_b", "parameters": {"param1": 3}}]}'

    result = parse_and_execute_json(functions, json_string)
    expected_result = {
        "results": {"mock_function_a": "3", "mock_function_b": "6"},
        "summary": "mock_function_a: 3\nmock_function_b: 6",
    }

    assert (
        result == expected_result
    ), f"Expected {expected_result}, but got {result}"


def test_parse_and_execute_json_function_not_found():
    functions = [mock_function_a]
    json_string = '{"functions": [{"name": "non_existent_function", "parameters": {}}]}'

    result = parse_and_execute_json(functions, json_string)
    expected_result = {
        "results": {
            "non_existent_function": "Error: Function non_existent_function not found"
        },
        "summary": "non_existent_function: Error: Function non_existent_function not found",
    }

    assert (
        result == expected_result
    ), f"Expected {expected_result}, but got {result}"


def test_parse_and_execute_json_error_handling():
    functions = [mock_function_b]
    json_string = '{"functions": [{"name": "mock_function_b", "parameters": {"param1": -1}}]}'

    result = parse_and_execute_json(functions, json_string)
    expected_result = {
        "results": {
            "mock_function_b": "Error: Negative value not allowed"
        },
        "summary": "mock_function_b: Error: Negative value not allowed",
    }

    assert (
        result == expected_result
    ), f"Expected {expected_result}, but got {result}"


# Run tests
run_test(
    "Test parse_and_execute_json success",
    test_parse_and_execute_json_success,
)
print("------------------------------------------------")
run_test(
    "Test parse_and_execute_json function not found",
    test_parse_and_execute_json_function_not_found,
)
print("------------------------------------------------")
run_test(
    "Test parse_and_execute_json error handling",
    test_parse_and_execute_json_error_handling,
)
