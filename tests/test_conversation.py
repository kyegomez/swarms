import os
from loguru import logger
from swarms.structs.conversation import Conversation


def assert_equal(actual, expected, message=""):
    """Custom assertion function for equality"""
    if actual != expected:
        logger.error(
            f"Assertion failed: {message}\nExpected: {expected}\nActual: {actual}"
        )
        raise AssertionError(
            f"{message}\nExpected: {expected}\nActual: {actual}"
        )
    logger.success(f"Assertion passed: {message}")


def assert_true(condition, message=""):
    """Custom assertion function for boolean conditions"""
    if not condition:
        logger.error(f"Assertion failed: {message}")
        raise AssertionError(message)
    logger.success(f"Assertion passed: {message}")


def test_conversation_initialization():
    """Test conversation initialization with different parameters"""
    logger.info("Testing conversation initialization")

    # Test default initialization
    conv = Conversation()
    assert_true(
        isinstance(conv, Conversation),
        "Should create Conversation instance",
    )
    assert_equal(
        conv.provider,
        "in-memory",
        "Default provider should be in-memory",
    )

    # Test with custom parameters
    conv = Conversation(
        name="test-conv",
        system_prompt="Test system prompt",
        time_enabled=True,
        token_count=True,
    )
    assert_equal(
        conv.name, "test-conv", "Name should be set correctly"
    )
    assert_equal(
        conv.system_prompt,
        "Test system prompt",
        "System prompt should be set",
    )
    assert_true(conv.time_enabled, "Time should be enabled")
    assert_true(conv.token_count, "Token count should be enabled")


def test_add_message():
    """Test adding messages to conversation"""
    logger.info("Testing add message functionality")

    conv = Conversation(time_enabled=True, token_count=True)

    # Test adding text message
    conv.add("user", "Hello, world!")
    assert_equal(
        len(conv.conversation_history), 1, "Should have one message"
    )
    assert_equal(
        conv.conversation_history[0]["role"],
        "user",
        "Role should be user",
    )
    assert_equal(
        conv.conversation_history[0]["content"],
        "Hello, world!",
        "Content should match",
    )

    # Test adding dict message
    dict_msg = {"key": "value"}
    conv.add("assistant", dict_msg)
    assert_equal(
        len(conv.conversation_history), 2, "Should have two messages"
    )
    assert_equal(
        conv.conversation_history[1]["role"],
        "assistant",
        "Role should be assistant",
    )
    assert_equal(
        conv.conversation_history[1]["content"],
        dict_msg,
        "Content should match dict",
    )


def test_delete_message():
    """Test deleting messages from conversation"""
    logger.info("Testing delete message functionality")

    conv = Conversation()
    conv.add("user", "Message 1")
    conv.add("user", "Message 2")

    initial_length = len(conv.conversation_history)
    conv.delete("0")  # Delete first message

    assert_equal(
        len(conv.conversation_history),
        initial_length - 1,
        "Conversation history should be shorter by one",
    )
    assert_equal(
        conv.conversation_history[0]["content"],
        "Message 2",
        "Remaining message should be Message 2",
    )


def test_update_message():
    """Test updating messages in conversation"""
    logger.info("Testing update message functionality")

    conv = Conversation()
    conv.add("user", "Original message")

    conv.update("0", "user", "Updated message")
    assert_equal(
        conv.conversation_history[0]["content"],
        "Updated message",
        "Message should be updated",
    )


def test_search_messages():
    """Test searching messages in conversation"""
    logger.info("Testing search functionality")

    conv = Conversation()
    conv.add("user", "Hello world")
    conv.add("assistant", "Hello user")
    conv.add("user", "Goodbye world")

    results = conv.search("Hello")
    assert_equal(
        len(results), 2, "Should find two messages with 'Hello'"
    )

    results = conv.search("Goodbye")
    assert_equal(
        len(results), 1, "Should find one message with 'Goodbye'"
    )


def test_export_import():
    """Test exporting and importing conversation"""
    logger.info("Testing export/import functionality")

    conv = Conversation(name="export-test")
    conv.add("user", "Test message")

    # Test JSON export/import
    test_file = "test_conversation_export.json"
    conv.export_conversation(test_file)

    assert_true(os.path.exists(test_file), "Export file should exist")

    new_conv = Conversation(name="import-test")
    new_conv.import_conversation(test_file)

    assert_equal(
        len(new_conv.conversation_history),
        len(conv.conversation_history),
        "Imported conversation should have same number of messages",
    )

    # Cleanup
    os.remove(test_file)


def test_message_counting():
    """Test message counting functionality"""
    logger.info("Testing message counting functionality")

    conv = Conversation()
    conv.add("user", "User message")
    conv.add("assistant", "Assistant message")
    conv.add("system", "System message")

    counts = conv.count_messages_by_role()
    assert_equal(counts["user"], 1, "Should have one user message")
    assert_equal(
        counts["assistant"], 1, "Should have one assistant message"
    )
    assert_equal(
        counts["system"], 1, "Should have one system message"
    )


def test_conversation_string_representation():
    """Test string representation methods"""
    logger.info("Testing string representation methods")

    conv = Conversation()
    conv.add("user", "Test message")

    str_repr = conv.return_history_as_string()
    assert_true(
        "user: Test message" in str_repr,
        "String representation should contain message",
    )

    json_repr = conv.to_json()
    assert_true(
        isinstance(json_repr, str),
        "JSON representation should be string",
    )
    assert_true(
        "Test message" in json_repr,
        "JSON should contain message content",
    )


def test_memory_management():
    """Test memory management functions"""
    logger.info("Testing memory management functions")

    conv = Conversation()
    conv.add("user", "Message 1")
    conv.add("assistant", "Message 2")

    # Test clear
    conv.clear()
    assert_equal(
        len(conv.conversation_history),
        0,
        "History should be empty after clear",
    )

    # Test truncate
    conv = Conversation(context_length=100, token_count=True)
    long_message = (
        "This is a very long message that should be truncated " * 10
    )
    conv.add("user", long_message)
    conv.truncate_memory_with_tokenizer()
    assert_true(
        len(conv.conversation_history[0]["content"])
        < len(long_message),
        "Message should be truncated",
    )


def test_backend_initialization():
    """Test different backend initializations"""
    logger.info("Testing backend initialization")

    # Test Redis backend
    conv = Conversation(
        backend="redis",
        redis_host="localhost",
        redis_port=6379,
        redis_db=0,
        use_embedded_redis=True,
    )
    assert_equal(conv.backend, "redis", "Backend should be redis")

    # Test SQLite backend
    conv = Conversation(
        backend="sqlite",
        db_path=":memory:",
        table_name="test_conversations",
    )
    assert_equal(conv.backend, "sqlite", "Backend should be sqlite")

    # Test DuckDB backend
    conv = Conversation(
        backend="duckdb",
        db_path=":memory:",
        table_name="test_conversations",
    )
    assert_equal(conv.backend, "duckdb", "Backend should be duckdb")


def test_conversation_with_system_prompt():
    """Test conversation with system prompt and rules"""
    logger.info("Testing conversation with system prompt and rules")

    conv = Conversation(
        system_prompt="You are a helpful assistant",
        rules="Be concise and clear",
        custom_rules_prompt="Follow these guidelines",
        time_enabled=True,
    )

    history = conv.conversation_history
    assert_equal(
        len(history),
        3,
        "Should have system prompt, rules, and custom rules",
    )
    assert_equal(
        history[0]["content"],
        "You are a helpful assistant",
        "System prompt should match",
    )
    assert_equal(
        history[1]["content"],
        "Be concise and clear",
        "Rules should match",
    )
    assert_true(
        "timestamp" in history[0], "Messages should have timestamps"
    )


def test_batch_operations():
    """Test batch operations on conversation"""
    logger.info("Testing batch operations")

    conv = Conversation()

    # Test batch add
    roles = ["user", "assistant", "user"]
    contents = ["Hello", "Hi there", "How are you?"]
    conv.add_multiple_messages(roles, contents)

    assert_equal(
        len(conv.conversation_history),
        3,
        "Should have three messages",
    )

    # Test batch search
    results = conv.search("Hi")
    assert_equal(len(results), 1, "Should find one message with 'Hi'")


def test_conversation_export_formats():
    """Test different export formats"""
    logger.info("Testing export formats")

    conv = Conversation(name="export-test")
    conv.add("user", "Test message")

    # Test YAML export
    conv.export_method = "yaml"
    conv.save_filepath = "test_conversation.yaml"
    conv.export()
    assert_true(
        os.path.exists("test_conversation.yaml"),
        "YAML file should exist",
    )

    # Test JSON export
    conv.export_method = "json"
    conv.save_filepath = "test_conversation.json"
    conv.export()
    assert_true(
        os.path.exists("test_conversation.json"),
        "JSON file should exist",
    )

    # Cleanup
    os.remove("test_conversation.yaml")
    os.remove("test_conversation.json")


def test_conversation_with_token_counting():
    """Test conversation with token counting enabled"""
    logger.info("Testing token counting functionality")

    conv = Conversation(
        token_count=True,
        tokenizer_model_name="gpt-4.1",
        context_length=1000,
    )

    conv.add("user", "This is a test message")
    assert_true(
        "token_count" in conv.conversation_history[0],
        "Message should have token count",
    )

    # Test token counting with different message types
    conv.add(
        "assistant", {"response": "This is a structured response"}
    )
    assert_true(
        "token_count" in conv.conversation_history[1],
        "Structured message should have token count",
    )


def test_conversation_message_categories():
    """Test conversation with message categories"""
    logger.info("Testing message categories")

    conv = Conversation()

    # Add messages with categories
    conv.add("user", "Input message", category="input")
    conv.add("assistant", "Output message", category="output")

    # Test category counting
    token_counts = conv.export_and_count_categories()
    assert_true(
        "input_tokens" in token_counts,
        "Should have input token count",
    )
    assert_true(
        "output_tokens" in token_counts,
        "Should have output token count",
    )
    assert_true(
        "total_tokens" in token_counts,
        "Should have total token count",
    )


def test_conversation_persistence():
    """Test conversation persistence and loading"""
    logger.info("Testing conversation persistence")

    # Create and save conversation
    conv1 = Conversation(
        name="persistence-test",
        system_prompt="Test prompt",
        time_enabled=True,
        autosave=True,
    )
    conv1.add("user", "Test message")
    conv1.export()

    # Load conversation
    conv2 = Conversation.load_conversation(name="persistence-test")
    assert_equal(
        conv2.system_prompt,
        "Test prompt",
        "System prompt should persist",
    )
    assert_equal(
        len(conv2.conversation_history),
        2,
        "Should have system prompt and message",
    )


def test_conversation_utilities():
    """Test various utility methods"""
    logger.info("Testing utility methods")

    conv = Conversation(message_id_on=True)
    conv.add("user", "First message")
    conv.add("assistant", "Second message")

    # Test getting last message
    last_msg = conv.get_last_message_as_string()
    assert_true(
        "Second message" in last_msg,
        "Should get correct last message",
    )

    # Test getting messages as list
    msg_list = conv.return_messages_as_list()
    assert_equal(len(msg_list), 2, "Should have two messages in list")

    # Test getting messages as dictionary
    msg_dict = conv.return_messages_as_dictionary()
    assert_equal(
        len(msg_dict), 2, "Should have two messages in dictionary"
    )

    # Test message IDs
    assert_true(
        "message_id" in conv.conversation_history[0],
        "Messages should have IDs when enabled",
    )


def test_conversation_error_handling():
    """Test error handling in conversation methods"""
    logger.info("Testing error handling")

    conv = Conversation()

    # Test invalid export method
    try:
        conv.export_method = "invalid"
        conv.export()
        assert_true(
            False, "Should raise ValueError for invalid export method"
        )
    except ValueError:
        assert_true(
            True, "Should catch ValueError for invalid export method"
        )

    # Test invalid backend
    try:
        Conversation(backend="invalid_backend")
        assert_true(
            False, "Should raise ValueError for invalid backend"
        )
    except ValueError:
        assert_true(
            True, "Should catch ValueError for invalid backend"
        )


def run_all_tests():
    """Run all test functions"""
    logger.info("Starting all tests")

    test_functions = [
        test_conversation_initialization,
        test_add_message,
        test_delete_message,
        test_update_message,
        test_search_messages,
        test_export_import,
        test_message_counting,
        test_conversation_string_representation,
        test_memory_management,
        test_backend_initialization,
        test_conversation_with_system_prompt,
        test_batch_operations,
        test_conversation_export_formats,
        test_conversation_with_token_counting,
        test_conversation_message_categories,
        test_conversation_persistence,
        test_conversation_utilities,
        test_conversation_error_handling,
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            logger.info(f"Running {test_func.__name__}")
            test_func()
            passed += 1
            logger.success(f"{test_func.__name__} passed")
        except Exception as e:
            failed += 1
            logger.error(f"{test_func.__name__} failed: {str(e)}")

    logger.info(f"Test summary: {passed} passed, {failed} failed")
    return passed, failed


if __name__ == "__main__":
    passed, failed = run_all_tests()
    if failed > 0:
        exit(1)
