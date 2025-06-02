import os
import sys
from pathlib import Path
import tempfile
import threading

# Add the project root to Python path to allow imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from swarms.communication.duckdb_wrap import (
    DuckDBConversation,
    Message,
    MessageType,
)


def setup_test():
    """Set up test environment."""
    temp_dir = tempfile.TemporaryDirectory()
    db_path = Path(temp_dir.name) / "test_conversations.duckdb"
    conversation = DuckDBConversation(
        db_path=str(db_path),
        enable_timestamps=True,
        enable_logging=True,
    )
    return temp_dir, db_path, conversation


def cleanup_test(temp_dir, db_path):
    """Clean up test environment."""
    if os.path.exists(db_path):
        os.remove(db_path)
    temp_dir.cleanup()


def test_initialization():
    """Test conversation initialization."""
    temp_dir, db_path, _ = setup_test()
    try:
        conv = DuckDBConversation(db_path=str(db_path))
        assert conv.db_path == db_path, "Database path mismatch"
        assert (
            conv.table_name == "conversations"
        ), "Table name mismatch"
        assert (
            conv.enable_timestamps is True
        ), "Timestamps should be enabled"
        assert (
            conv.current_conversation_id is not None
        ), "Conversation ID should not be None"
        print("✓ Initialization test passed")
    finally:
        cleanup_test(temp_dir, db_path)


def test_add_message():
    """Test adding a single message."""
    temp_dir, db_path, conversation = setup_test()
    try:
        msg_id = conversation.add(
            role="user",
            content="Hello, world!",
            message_type=MessageType.USER,
        )
        assert msg_id is not None, "Message ID should not be None"
        assert isinstance(
            msg_id, int
        ), "Message ID should be an integer"
        print("✓ Add message test passed")
    finally:
        cleanup_test(temp_dir, db_path)


def test_add_complex_message():
    """Test adding a message with complex content."""
    temp_dir, db_path, conversation = setup_test()
    try:
        complex_content = {
            "text": "Hello",
            "data": [1, 2, 3],
            "nested": {"key": "value"},
        }
        msg_id = conversation.add(
            role="assistant",
            content=complex_content,
            message_type=MessageType.ASSISTANT,
            metadata={"source": "test"},
            token_count=10,
        )
        assert msg_id is not None, "Message ID should not be None"
        print("✓ Add complex message test passed")
    finally:
        cleanup_test(temp_dir, db_path)


def test_batch_add():
    """Test batch adding messages."""
    temp_dir, db_path, conversation = setup_test()
    try:
        messages = [
            Message(
                role="user",
                content="First message",
                message_type=MessageType.USER,
            ),
            Message(
                role="assistant",
                content="Second message",
                message_type=MessageType.ASSISTANT,
            ),
        ]
        msg_ids = conversation.batch_add(messages)
        assert len(msg_ids) == 2, "Should have 2 message IDs"
        assert all(
            isinstance(id, int) for id in msg_ids
        ), "All IDs should be integers"
        print("✓ Batch add test passed")
    finally:
        cleanup_test(temp_dir, db_path)


def test_get_str():
    """Test getting conversation as string."""
    temp_dir, db_path, conversation = setup_test()
    try:
        conversation.add("user", "Hello")
        conversation.add("assistant", "Hi there!")
        conv_str = conversation.get_str()
        assert "user: Hello" in conv_str, "User message not found"
        assert (
            "assistant: Hi there!" in conv_str
        ), "Assistant message not found"
        print("✓ Get string test passed")
    finally:
        cleanup_test(temp_dir, db_path)


def test_get_messages():
    """Test getting messages with pagination."""
    temp_dir, db_path, conversation = setup_test()
    try:
        for i in range(5):
            conversation.add("user", f"Message {i}")

        all_messages = conversation.get_messages()
        assert len(all_messages) == 5, "Should have 5 messages"

        limited_messages = conversation.get_messages(limit=2)
        assert (
            len(limited_messages) == 2
        ), "Should have 2 limited messages"

        offset_messages = conversation.get_messages(offset=2)
        assert (
            len(offset_messages) == 3
        ), "Should have 3 offset messages"
        print("✓ Get messages test passed")
    finally:
        cleanup_test(temp_dir, db_path)


def test_search_messages():
    """Test searching messages."""
    temp_dir, db_path, conversation = setup_test()
    try:
        conversation.add("user", "Hello world")
        conversation.add("assistant", "Hello there")
        conversation.add("user", "Goodbye world")

        results = conversation.search_messages("world")
        assert (
            len(results) == 2
        ), "Should find 2 messages with 'world'"
        assert all(
            "world" in msg["content"] for msg in results
        ), "All results should contain 'world'"
        print("✓ Search messages test passed")
    finally:
        cleanup_test(temp_dir, db_path)


def test_get_statistics():
    """Test getting conversation statistics."""
    temp_dir, db_path, conversation = setup_test()
    try:
        conversation.add("user", "Hello", token_count=2)
        conversation.add("assistant", "Hi", token_count=1)

        stats = conversation.get_statistics()
        assert (
            stats["total_messages"] == 2
        ), "Should have 2 total messages"
        assert (
            stats["unique_roles"] == 2
        ), "Should have 2 unique roles"
        assert (
            stats["total_tokens"] == 3
        ), "Should have 3 total tokens"
        print("✓ Get statistics test passed")
    finally:
        cleanup_test(temp_dir, db_path)


def test_json_operations():
    """Test JSON save and load operations."""
    temp_dir, db_path, conversation = setup_test()
    try:
        conversation.add("user", "Hello")
        conversation.add("assistant", "Hi")

        json_path = Path(temp_dir.name) / "test_conversation.json"
        conversation.save_as_json(str(json_path))
        assert json_path.exists(), "JSON file should exist"

        new_conversation = DuckDBConversation(
            db_path=str(Path(temp_dir.name) / "new.duckdb")
        )
        assert new_conversation.load_from_json(
            str(json_path)
        ), "Should load from JSON"
        assert (
            len(new_conversation.get_messages()) == 2
        ), "Should have 2 messages after load"
        print("✓ JSON operations test passed")
    finally:
        cleanup_test(temp_dir, db_path)


def test_yaml_operations():
    """Test YAML save and load operations."""
    temp_dir, db_path, conversation = setup_test()
    try:
        conversation.add("user", "Hello")
        conversation.add("assistant", "Hi")

        yaml_path = Path(temp_dir.name) / "test_conversation.yaml"
        conversation.save_as_yaml(str(yaml_path))
        assert yaml_path.exists(), "YAML file should exist"

        new_conversation = DuckDBConversation(
            db_path=str(Path(temp_dir.name) / "new.duckdb")
        )
        assert new_conversation.load_from_yaml(
            str(yaml_path)
        ), "Should load from YAML"
        assert (
            len(new_conversation.get_messages()) == 2
        ), "Should have 2 messages after load"
        print("✓ YAML operations test passed")
    finally:
        cleanup_test(temp_dir, db_path)


def test_message_types():
    """Test different message types."""
    temp_dir, db_path, conversation = setup_test()
    try:
        conversation.add(
            "system",
            "System message",
            message_type=MessageType.SYSTEM,
        )
        conversation.add(
            "user", "User message", message_type=MessageType.USER
        )
        conversation.add(
            "assistant",
            "Assistant message",
            message_type=MessageType.ASSISTANT,
        )
        conversation.add(
            "function",
            "Function message",
            message_type=MessageType.FUNCTION,
        )
        conversation.add(
            "tool", "Tool message", message_type=MessageType.TOOL
        )

        messages = conversation.get_messages()
        assert len(messages) == 5, "Should have 5 messages"
        assert all(
            "message_type" in msg for msg in messages
        ), "All messages should have type"
        print("✓ Message types test passed")
    finally:
        cleanup_test(temp_dir, db_path)


def test_delete_operations():
    """Test deletion operations."""
    temp_dir, db_path, conversation = setup_test()
    try:
        conversation.add("user", "Hello")
        conversation.add("assistant", "Hi")

        assert (
            conversation.delete_current_conversation()
        ), "Should delete conversation"
        assert (
            len(conversation.get_messages()) == 0
        ), "Should have no messages after delete"

        conversation.add("user", "New message")
        assert conversation.clear_all(), "Should clear all messages"
        assert (
            len(conversation.get_messages()) == 0
        ), "Should have no messages after clear"
        print("✓ Delete operations test passed")
    finally:
        cleanup_test(temp_dir, db_path)


def test_concurrent_operations():
    """Test concurrent operations."""
    temp_dir, db_path, conversation = setup_test()
    try:

        def add_messages():
            for i in range(10):
                conversation.add("user", f"Message {i}")

        threads = [
            threading.Thread(target=add_messages) for _ in range(5)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        messages = conversation.get_messages()
        assert (
            len(messages) == 50
        ), "Should have 50 messages (10 * 5 threads)"
        print("✓ Concurrent operations test passed")
    finally:
        cleanup_test(temp_dir, db_path)


def test_error_handling():
    """Test error handling."""
    temp_dir, db_path, conversation = setup_test()
    try:
        # Test invalid message type
        try:
            conversation.add(
                "user", "Message", message_type="invalid"
            )
            assert (
                False
            ), "Should raise exception for invalid message type"
        except Exception:
            pass

        # Test invalid JSON content
        try:
            conversation.add("user", {"invalid": object()})
            assert (
                False
            ), "Should raise exception for invalid JSON content"
        except Exception:
            pass

        # Test invalid file operations
        try:
            conversation.load_from_json("/nonexistent/path.json")
            assert (
                False
            ), "Should raise exception for invalid file path"
        except Exception:
            pass

        print("✓ Error handling test passed")
    finally:
        cleanup_test(temp_dir, db_path)


def run_all_tests():
    """Run all tests."""
    print("Running DuckDB Conversation tests...")
    tests = [
        test_initialization,
        test_add_message,
        test_add_complex_message,
        test_batch_add,
        test_get_str,
        test_get_messages,
        test_search_messages,
        test_get_statistics,
        test_json_operations,
        test_yaml_operations,
        test_message_types,
        test_delete_operations,
        test_concurrent_operations,
        test_error_handling,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} failed: {str(e)}")
            raise

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    run_all_tests()
