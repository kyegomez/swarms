import shutil
from datetime import datetime
from pathlib import Path

from loguru import logger

from swarms.structs.conversation import Conversation


def setup_temp_conversations_dir():
    """Create a temporary directory for conversation cache files."""
    temp_dir = Path("temp_test_conversations")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    logger.info(f"Created temporary test directory: {temp_dir}")
    return temp_dir


def create_test_conversation(temp_dir):
    """Create a basic conversation for testing."""
    conv = Conversation(
        name="test_conversation", conversations_dir=str(temp_dir)
    )
    conv.add("user", "Hello, world!")
    conv.add("assistant", "Hello, user!")
    logger.info("Created test conversation with basic messages")
    return conv


def test_add_message():
    logger.info("Running test_add_message")
    conv = Conversation()
    conv.add("user", "Hello, world!")
    try:
        assert len(conv.conversation_history) == 1
        assert conv.conversation_history[0]["role"] == "user"
        assert (
            conv.conversation_history[0]["content"] == "Hello, world!"
        )
        logger.success("test_add_message passed")
        return True
    except AssertionError as e:
        logger.error(f"test_add_message failed: {str(e)}")
        return False


def test_add_message_with_time():
    logger.info("Running test_add_message_with_time")
    conv = Conversation(time_enabled=False)
    conv.add("user", "Hello, world!")
    try:
        assert len(conv.conversation_history) == 1
        assert conv.conversation_history[0]["role"] == "user"
        assert (
            conv.conversation_history[0]["content"] == "Hello, world!"
        )
        assert "timestamp" in conv.conversation_history[0]
        logger.success("test_add_message_with_time passed")
        return True
    except AssertionError as e:
        logger.error(f"test_add_message_with_time failed: {str(e)}")
        return False


def test_delete_message():
    logger.info("Running test_delete_message")
    conv = Conversation()
    conv.add("user", "Hello, world!")
    conv.delete(0)
    try:
        assert len(conv.conversation_history) == 0
        logger.success("test_delete_message passed")
        return True
    except AssertionError as e:
        logger.error(f"test_delete_message failed: {str(e)}")
        return False


def test_delete_message_out_of_bounds():
    logger.info("Running test_delete_message_out_of_bounds")
    conv = Conversation()
    conv.add("user", "Hello, world!")
    try:
        conv.delete(1)
        logger.error(
            "test_delete_message_out_of_bounds failed: Expected IndexError"
        )
        return False
    except IndexError:
        logger.success("test_delete_message_out_of_bounds passed")
        return True


def test_update_message():
    logger.info("Running test_update_message")
    conv = Conversation()
    conv.add("user", "Hello, world!")
    conv.update(0, "assistant", "Hello, user!")
    try:
        assert len(conv.conversation_history) == 1
        assert conv.conversation_history[0]["role"] == "assistant"
        assert (
            conv.conversation_history[0]["content"] == "Hello, user!"
        )
        logger.success("test_update_message passed")
        return True
    except AssertionError as e:
        logger.error(f"test_update_message failed: {str(e)}")
        return False


def test_update_message_out_of_bounds():
    logger.info("Running test_update_message_out_of_bounds")
    conv = Conversation()
    conv.add("user", "Hello, world!")
    try:
        conv.update(1, "assistant", "Hello, user!")
        logger.error(
            "test_update_message_out_of_bounds failed: Expected IndexError"
        )
        return False
    except IndexError:
        logger.success("test_update_message_out_of_bounds passed")
        return True


def test_return_history_as_string():
    logger.info("Running test_return_history_as_string")
    conv = Conversation()
    conv.add("user", "Hello, world!")
    conv.add("assistant", "Hello, user!")
    result = conv.return_history_as_string()
    expected = "user: Hello, world!\n\nassistant: Hello, user!\n\n"
    try:
        assert result == expected
        logger.success("test_return_history_as_string passed")
        return True
    except AssertionError as e:
        logger.error(
            f"test_return_history_as_string failed: {str(e)}"
        )
        return False


def test_search():
    logger.info("Running test_search")
    conv = Conversation()
    conv.add("user", "Hello, world!")
    conv.add("assistant", "Hello, user!")
    results = conv.search("Hello")
    try:
        assert len(results) == 2
        assert results[0]["content"] == "Hello, world!"
        assert results[1]["content"] == "Hello, user!"
        logger.success("test_search passed")
        return True
    except AssertionError as e:
        logger.error(f"test_search failed: {str(e)}")
        return False


def test_conversation_cache_creation():
    logger.info("Running test_conversation_cache_creation")
    temp_dir = setup_temp_conversations_dir()
    try:
        conv = Conversation(
            name="cache_test", conversations_dir=str(temp_dir)
        )
        conv.add("user", "Test message")
        cache_file = temp_dir / "cache_test.json"
        result = cache_file.exists()
        if result:
            logger.success("test_conversation_cache_creation passed")
        else:
            logger.error(
                "test_conversation_cache_creation failed: Cache file not created"
            )
        return result
    finally:
        shutil.rmtree(temp_dir)


def test_conversation_cache_loading():
    logger.info("Running test_conversation_cache_loading")
    temp_dir = setup_temp_conversations_dir()
    try:
        conv1 = Conversation(
            name="load_test", conversations_dir=str(temp_dir)
        )
        conv1.add("user", "Test message")

        conv2 = Conversation.load_conversation(
            name="load_test", conversations_dir=str(temp_dir)
        )
        result = (
            len(conv2.conversation_history) == 1
            and conv2.conversation_history[0]["content"]
            == "Test message"
        )
        if result:
            logger.success("test_conversation_cache_loading passed")
        else:
            logger.error(
                "test_conversation_cache_loading failed: Loaded conversation mismatch"
            )
        return result
    finally:
        shutil.rmtree(temp_dir)


def test_add_multiple_messages():
    logger.info("Running test_add_multiple_messages")
    conv = Conversation()
    roles = ["user", "assistant", "system"]
    contents = ["Hello", "Hi there", "System message"]
    conv.add_multiple_messages(roles, contents)
    try:
        assert len(conv.conversation_history) == 3
        assert conv.conversation_history[0]["role"] == "user"
        assert conv.conversation_history[1]["role"] == "assistant"
        assert conv.conversation_history[2]["role"] == "system"
        logger.success("test_add_multiple_messages passed")
        return True
    except AssertionError as e:
        logger.error(f"test_add_multiple_messages failed: {str(e)}")
        return False


def test_query():
    logger.info("Running test_query")
    conv = Conversation()
    conv.add("user", "Test message")
    try:
        result = conv.query(0)
        assert result["role"] == "user"
        assert result["content"] == "Test message"
        logger.success("test_query passed")
        return True
    except AssertionError as e:
        logger.error(f"test_query failed: {str(e)}")
        return False


def test_display_conversation():
    logger.info("Running test_display_conversation")
    conv = Conversation()
    conv.add("user", "Hello")
    conv.add("assistant", "Hi")
    try:
        conv.display_conversation()
        logger.success("test_display_conversation passed")
        return True
    except Exception as e:
        logger.error(f"test_display_conversation failed: {str(e)}")
        return False


def test_count_messages_by_role():
    logger.info("Running test_count_messages_by_role")
    conv = Conversation()
    conv.add("user", "Hello")
    conv.add("assistant", "Hi")
    conv.add("system", "System message")
    try:
        counts = conv.count_messages_by_role()
        assert counts["user"] == 1
        assert counts["assistant"] == 1
        assert counts["system"] == 1
        logger.success("test_count_messages_by_role passed")
        return True
    except AssertionError as e:
        logger.error(f"test_count_messages_by_role failed: {str(e)}")
        return False


def test_get_str():
    logger.info("Running test_get_str")
    conv = Conversation()
    conv.add("user", "Hello")
    try:
        result = conv.get_str()
        assert "user: Hello" in result
        logger.success("test_get_str passed")
        return True
    except AssertionError as e:
        logger.error(f"test_get_str failed: {str(e)}")
        return False


def test_to_json():
    logger.info("Running test_to_json")
    conv = Conversation()
    conv.add("user", "Hello")
    try:
        result = conv.to_json()
        assert isinstance(result, str)
        assert "Hello" in result
        logger.success("test_to_json passed")
        return True
    except AssertionError as e:
        logger.error(f"test_to_json failed: {str(e)}")
        return False


def test_to_dict():
    logger.info("Running test_to_dict")
    conv = Conversation()
    conv.add("user", "Hello")
    try:
        result = conv.to_dict()
        assert isinstance(result, list)
        assert result[0]["content"] == "Hello"
        logger.success("test_to_dict passed")
        return True
    except AssertionError as e:
        logger.error(f"test_to_dict failed: {str(e)}")
        return False


def test_to_yaml():
    logger.info("Running test_to_yaml")
    conv = Conversation()
    conv.add("user", "Hello")
    try:
        result = conv.to_yaml()
        assert isinstance(result, str)
        assert "Hello" in result
        logger.success("test_to_yaml passed")
        return True
    except AssertionError as e:
        logger.error(f"test_to_yaml failed: {str(e)}")
        return False


def test_get_last_message_as_string():
    logger.info("Running test_get_last_message_as_string")
    conv = Conversation()
    conv.add("user", "First")
    conv.add("assistant", "Last")
    try:
        result = conv.get_last_message_as_string()
        assert result == "assistant: Last"
        logger.success("test_get_last_message_as_string passed")
        return True
    except AssertionError as e:
        logger.error(
            f"test_get_last_message_as_string failed: {str(e)}"
        )
        return False


def test_return_messages_as_list():
    logger.info("Running test_return_messages_as_list")
    conv = Conversation()
    conv.add("user", "Hello")
    conv.add("assistant", "Hi")
    try:
        result = conv.return_messages_as_list()
        assert len(result) == 2
        assert result[0] == "user: Hello"
        assert result[1] == "assistant: Hi"
        logger.success("test_return_messages_as_list passed")
        return True
    except AssertionError as e:
        logger.error(f"test_return_messages_as_list failed: {str(e)}")
        return False


def test_return_messages_as_dictionary():
    logger.info("Running test_return_messages_as_dictionary")
    conv = Conversation()
    conv.add("user", "Hello")
    try:
        result = conv.return_messages_as_dictionary()
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"
        logger.success("test_return_messages_as_dictionary passed")
        return True
    except AssertionError as e:
        logger.error(
            f"test_return_messages_as_dictionary failed: {str(e)}"
        )
        return False


def test_add_tool_output_to_agent():
    logger.info("Running test_add_tool_output_to_agent")
    conv = Conversation()
    tool_output = {"name": "test_tool", "output": "test result"}
    try:
        conv.add_tool_output_to_agent("tool", tool_output)
        assert len(conv.conversation_history) == 1
        assert conv.conversation_history[0]["role"] == "tool"
        assert conv.conversation_history[0]["content"] == tool_output
        logger.success("test_add_tool_output_to_agent passed")
        return True
    except AssertionError as e:
        logger.error(
            f"test_add_tool_output_to_agent failed: {str(e)}"
        )
        return False


def test_get_final_message():
    logger.info("Running test_get_final_message")
    conv = Conversation()
    conv.add("user", "First")
    conv.add("assistant", "Last")
    try:
        result = conv.get_final_message()
        assert result == "assistant: Last"
        logger.success("test_get_final_message passed")
        return True
    except AssertionError as e:
        logger.error(f"test_get_final_message failed: {str(e)}")
        return False


def test_get_final_message_content():
    logger.info("Running test_get_final_message_content")
    conv = Conversation()
    conv.add("user", "First")
    conv.add("assistant", "Last")
    try:
        result = conv.get_final_message_content()
        assert result == "Last"
        logger.success("test_get_final_message_content passed")
        return True
    except AssertionError as e:
        logger.error(
            f"test_get_final_message_content failed: {str(e)}"
        )
        return False


def test_return_all_except_first():
    logger.info("Running test_return_all_except_first")
    conv = Conversation()
    conv.add("system", "System")
    conv.add("user", "Hello")
    conv.add("assistant", "Hi")
    try:
        result = conv.return_all_except_first()
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        logger.success("test_return_all_except_first passed")
        return True
    except AssertionError as e:
        logger.error(f"test_return_all_except_first failed: {str(e)}")
        return False


def test_return_all_except_first_string():
    logger.info("Running test_return_all_except_first_string")
    conv = Conversation()
    conv.add("system", "System")
    conv.add("user", "Hello")
    conv.add("assistant", "Hi")
    try:
        result = conv.return_all_except_first_string()
        assert "Hello" in result
        assert "Hi" in result
        assert "System" not in result
        logger.success("test_return_all_except_first_string passed")
        return True
    except AssertionError as e:
        logger.error(
            f"test_return_all_except_first_string failed: {str(e)}"
        )
        return False


def test_batch_add():
    logger.info("Running test_batch_add")
    conv = Conversation()
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    try:
        conv.batch_add(messages)
        assert len(conv.conversation_history) == 2
        assert conv.conversation_history[0]["role"] == "user"
        assert conv.conversation_history[1]["role"] == "assistant"
        logger.success("test_batch_add passed")
        return True
    except AssertionError as e:
        logger.error(f"test_batch_add failed: {str(e)}")
        return False


def test_get_cache_stats():
    logger.info("Running test_get_cache_stats")
    conv = Conversation(cache_enabled=True)
    conv.add("user", "Hello")
    try:
        stats = conv.get_cache_stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "cached_tokens" in stats
        assert "total_tokens" in stats
        assert "hit_rate" in stats
        logger.success("test_get_cache_stats passed")
        return True
    except AssertionError as e:
        logger.error(f"test_get_cache_stats failed: {str(e)}")
        return False


def test_list_cached_conversations():
    logger.info("Running test_list_cached_conversations")
    temp_dir = setup_temp_conversations_dir()
    try:
        conv = Conversation(
            name="test_list", conversations_dir=str(temp_dir)
        )
        conv.add("user", "Test message")

        conversations = Conversation.list_cached_conversations(
            str(temp_dir)
        )
        try:
            assert "test_list" in conversations
            logger.success("test_list_cached_conversations passed")
            return True
        except AssertionError as e:
            logger.error(
                f"test_list_cached_conversations failed: {str(e)}"
            )
            return False
    finally:
        shutil.rmtree(temp_dir)


def test_clear():
    logger.info("Running test_clear")
    conv = Conversation()
    conv.add("user", "Hello")
    conv.add("assistant", "Hi")
    try:
        conv.clear()
        assert len(conv.conversation_history) == 0
        logger.success("test_clear passed")
        return True
    except AssertionError as e:
        logger.error(f"test_clear failed: {str(e)}")
        return False


def test_save_and_load_json():
    logger.info("Running test_save_and_load_json")
    temp_dir = setup_temp_conversations_dir()
    file_path = temp_dir / "test_save.json"

    try:
        conv = Conversation()
        conv.add("user", "Hello")
        conv.save_as_json(str(file_path))

        conv2 = Conversation()
        conv2.load_from_json(str(file_path))

        try:
            assert len(conv2.conversation_history) == 1
            assert conv2.conversation_history[0]["content"] == "Hello"
            logger.success("test_save_and_load_json passed")
            return True
        except AssertionError as e:
            logger.error(f"test_save_and_load_json failed: {str(e)}")
            return False
    finally:
        shutil.rmtree(temp_dir)


def run_all_tests():
    """Run all test functions and return results."""
    logger.info("Starting test suite execution")
    test_results = []
    test_functions = [
        test_add_message,
        test_add_message_with_time,
        test_delete_message,
        test_delete_message_out_of_bounds,
        test_update_message,
        test_update_message_out_of_bounds,
        test_return_history_as_string,
        test_search,
        test_conversation_cache_creation,
        test_conversation_cache_loading,
        test_add_multiple_messages,
        test_query,
        test_display_conversation,
        test_count_messages_by_role,
        test_get_str,
        test_to_json,
        test_to_dict,
        test_to_yaml,
        test_get_last_message_as_string,
        test_return_messages_as_list,
        test_return_messages_as_dictionary,
        test_add_tool_output_to_agent,
        test_get_final_message,
        test_get_final_message_content,
        test_return_all_except_first,
        test_return_all_except_first_string,
        test_batch_add,
        test_get_cache_stats,
        test_list_cached_conversations,
        test_clear,
        test_save_and_load_json,
    ]

    for test_func in test_functions:
        start_time = datetime.now()
        try:
            result = test_func()
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            test_results.append(
                {
                    "name": test_func.__name__,
                    "result": "PASS" if result else "FAIL",
                    "duration": duration,
                }
            )
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            test_results.append(
                {
                    "name": test_func.__name__,
                    "result": "ERROR",
                    "error": str(e),
                    "duration": duration,
                }
            )
            logger.error(
                f"Test {test_func.__name__} failed with error: {str(e)}"
            )

    return test_results


def generate_markdown_report(results):
    """Generate a markdown report from test results."""
    logger.info("Generating test report")

    # Summary
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["result"] == "PASS")
    failed_tests = sum(1 for r in results if r["result"] == "FAIL")
    error_tests = sum(1 for r in results if r["result"] == "ERROR")

    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {failed_tests}")
    logger.info(f"Errors: {error_tests}")

    report = "# Test Results Report\n\n"
    report += f"Test Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    report += "## Summary\n\n"
    report += f"- Total Tests: {total_tests}\n"
    report += f"- Passed: {passed_tests}\n"
    report += f"- Failed: {failed_tests}\n"
    report += f"- Errors: {error_tests}\n\n"

    # Detailed Results
    report += "## Detailed Results\n\n"
    report += "| Test Name | Result | Duration (s) | Error |\n"
    report += "|-----------|---------|--------------|-------|\n"

    for result in results:
        name = result["name"]
        test_result = result["result"]
        duration = f"{result['duration']:.4f}"
        error = result.get("error", "")
        report += (
            f"| {name} | {test_result} | {duration} | {error} |\n"
        )

    return report


if __name__ == "__main__":
    logger.info("Starting test execution")
    results = run_all_tests()
    report = generate_markdown_report(results)

    # Save report to file
    with open("test_results.md", "w") as f:
        f.write(report)

    logger.success(
        "Test execution completed. Results saved to test_results.md"
    )
