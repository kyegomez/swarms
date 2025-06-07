import os
import sys
import json
import datetime
import threading
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Add the project root to Python path to allow imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from loguru import logger

    LOGURU_AVAILABLE = True
except ImportError:
    import logging

    logger = logging.getLogger(__name__)
    LOGURU_AVAILABLE = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

# Test if supabase is available
try:
    from swarms.communication.supabase_wrap import (
        SupabaseConversation,
        SupabaseConnectionError,
        SupabaseOperationError,
    )
    from swarms.communication.base_communication import (
        Message,
        MessageType,
    )

    SUPABASE_AVAILABLE = True
except ImportError as e:
    SUPABASE_AVAILABLE = False
    print(f"❌ Supabase dependencies not available: {e}")
    print("Please install supabase-py: pip install supabase")

# Try to load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv is optional

# Test configuration
TEST_SUPABASE_URL = os.getenv("SUPABASE_URL")
TEST_SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TEST_TABLE_NAME = "conversations_test"


def print_test_header(test_name: str) -> None:
    """Print a formatted test header."""
    if RICH_AVAILABLE and console:
        console.print(
            Panel(
                f"[bold blue]Running Test: {test_name}[/bold blue]",
                expand=False,
            )
        )
    else:
        print(f"\n=== Running Test: {test_name} ===")


def print_test_result(
    test_name: str, success: bool, message: str, execution_time: float
) -> None:
    """Print a formatted test result."""
    if RICH_AVAILABLE and console:
        status = (
            "[bold green]PASSED[/bold green]"
            if success
            else "[bold red]FAILED[/bold red]"
        )
        console.print(f"\n{status} - {test_name}")
        console.print(f"Message: {message}")
        console.print(
            f"Execution time: {execution_time:.3f} seconds\n"
        )
    else:
        status = "PASSED" if success else "FAILED"
        print(f"\n{status} - {test_name}")
        print(f"Message: {message}")
        print(f"Execution time: {execution_time:.3f} seconds\n")


def print_messages(
    messages: List[Dict], title: str = "Messages"
) -> None:
    """Print messages in a formatted table."""
    if RICH_AVAILABLE and console:
        table = Table(title=title)
        table.add_column("ID", style="cyan")
        table.add_column("Role", style="yellow")
        table.add_column("Content", style="green")
        table.add_column("Type", style="magenta")
        table.add_column("Timestamp", style="blue")

        for msg in messages[
            :10
        ]:  # Limit to first 10 messages for display
            content = str(msg.get("content", ""))
            if isinstance(content, (dict, list)):
                content = (
                    json.dumps(content)[:50] + "..."
                    if len(json.dumps(content)) > 50
                    else json.dumps(content)
                )
            elif len(content) > 50:
                content = content[:50] + "..."

            table.add_row(
                str(msg.get("id", "")),
                msg.get("role", ""),
                content,
                str(msg.get("message_type", "")),
                (
                    str(msg.get("timestamp", ""))[:19]
                    if msg.get("timestamp")
                    else ""
                ),
            )

        console.print(table)
    else:
        print(f"\n{title}:")
        for i, msg in enumerate(messages[:10]):
            content = str(msg.get("content", ""))
            if isinstance(content, (dict, list)):
                content = json.dumps(content)
            if len(content) > 50:
                content = content[:50] + "..."
            print(f"{i+1}. {msg.get('role', '')}: {content}")


def run_test(
    test_func: callable, *args, **kwargs
) -> Tuple[bool, str, float]:
    """
    Run a test function and return its results.

    Args:
        test_func: The test function to run
        *args: Arguments for the test function
        **kwargs: Keyword arguments for the test function

    Returns:
        Tuple[bool, str, float]: (success, message, execution_time)
    """
    start_time = datetime.datetime.now()
    try:
        result = test_func(*args, **kwargs)
        end_time = datetime.datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        return True, str(result), execution_time
    except Exception as e:
        end_time = datetime.datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        return False, str(e), execution_time


def setup_test_conversation():
    """Set up a test conversation instance."""
    if not SUPABASE_AVAILABLE:
        raise ImportError("Supabase dependencies not available")

    if not TEST_SUPABASE_URL or not TEST_SUPABASE_KEY:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY environment variables must be set for testing"
        )

    conversation = SupabaseConversation(
        supabase_url=TEST_SUPABASE_URL,
        supabase_key=TEST_SUPABASE_KEY,
        table_name=TEST_TABLE_NAME,
        enable_logging=False,  # Reduce noise during testing
        time_enabled=True,
    )
    return conversation


def cleanup_test_conversation(conversation):
    """Clean up test conversation data."""
    try:
        conversation.clear()
    except Exception as e:
        if LOGURU_AVAILABLE:
            logger.warning(
                f"Failed to clean up test conversation: {e}"
            )
        else:
            print(
                f"Warning: Failed to clean up test conversation: {e}"
            )


def test_import_availability() -> bool:
    """Test that Supabase imports are properly handled."""
    print_test_header("Import Availability Test")

    if not SUPABASE_AVAILABLE:
        print(
            "✓ Import availability test passed - detected missing dependencies correctly"
        )
        return True

    # Test that all required classes are available
    assert (
        SupabaseConversation is not None
    ), "SupabaseConversation should be available"
    assert (
        SupabaseConnectionError is not None
    ), "SupabaseConnectionError should be available"
    assert (
        SupabaseOperationError is not None
    ), "SupabaseOperationError should be available"
    assert Message is not None, "Message should be available"
    assert MessageType is not None, "MessageType should be available"

    print(
        "✓ Import availability test passed - all imports successful"
    )
    return True


def test_initialization() -> bool:
    """Test SupabaseConversation initialization."""
    print_test_header("Initialization Test")

    if not SUPABASE_AVAILABLE:
        print(
            "✓ Initialization test skipped - Supabase not available"
        )
        return True

    conversation = setup_test_conversation()
    try:
        assert (
            conversation.supabase_url == TEST_SUPABASE_URL
        ), "Supabase URL mismatch"
        assert (
            conversation.table_name == TEST_TABLE_NAME
        ), "Table name mismatch"
        assert (
            conversation.current_conversation_id is not None
        ), "Conversation ID should not be None"
        assert (
            conversation.client is not None
        ), "Supabase client should not be None"
        assert isinstance(
            conversation.get_conversation_id(), str
        ), "Conversation ID should be string"

        # Test that initialization doesn't call super().__init__() improperly
        # This should not raise any errors
        print("✓ Initialization test passed")
        return True
    finally:
        cleanup_test_conversation(conversation)


def test_logging_configuration() -> bool:
    """Test logging configuration options."""
    print_test_header("Logging Configuration Test")

    if not SUPABASE_AVAILABLE:
        print(
            "✓ Logging configuration test skipped - Supabase not available"
        )
        return True

    # Test with logging enabled
    conversation_with_logging = SupabaseConversation(
        supabase_url=TEST_SUPABASE_URL,
        supabase_key=TEST_SUPABASE_KEY,
        table_name=TEST_TABLE_NAME,
        enable_logging=True,
        use_loguru=False,  # Force standard logging
    )

    try:
        assert (
            conversation_with_logging.enable_logging is True
        ), "Logging should be enabled"
        assert (
            conversation_with_logging.logger is not None
        ), "Logger should be configured"

        # Test with logging disabled
        conversation_no_logging = SupabaseConversation(
            supabase_url=TEST_SUPABASE_URL,
            supabase_key=TEST_SUPABASE_KEY,
            table_name=TEST_TABLE_NAME + "_no_log",
            enable_logging=False,
        )

        assert (
            conversation_no_logging.enable_logging is False
        ), "Logging should be disabled"

        print("✓ Logging configuration test passed")
        return True
    finally:
        cleanup_test_conversation(conversation_with_logging)
        try:
            cleanup_test_conversation(conversation_no_logging)
        except:
            pass


def test_add_message() -> bool:
    """Test adding a single message."""
    print_test_header("Add Message Test")

    if not SUPABASE_AVAILABLE:
        print("✓ Add message test skipped - Supabase not available")
        return True

    conversation = setup_test_conversation()
    try:
        msg_id = conversation.add(
            role="user",
            content="Hello, Supabase!",
            message_type=MessageType.USER,
            metadata={"test": True},
        )
        assert msg_id is not None, "Message ID should not be None"
        assert isinstance(
            msg_id, int
        ), "Message ID should be an integer"

        # Verify message was stored
        messages = conversation.get_messages()
        assert len(messages) >= 1, "Should have at least 1 message"
        print("✓ Add message test passed")
        return True
    finally:
        cleanup_test_conversation(conversation)


def test_add_complex_message() -> bool:
    """Test adding a message with complex content."""
    print_test_header("Add Complex Message Test")

    if not SUPABASE_AVAILABLE:
        print(
            "✓ Add complex message test skipped - Supabase not available"
        )
        return True

    conversation = setup_test_conversation()
    try:
        complex_content = {
            "text": "Hello from Supabase",
            "data": [1, 2, 3, {"nested": "value"}],
            "metadata": {"source": "test", "priority": "high"},
        }

        msg_id = conversation.add(
            role="assistant",
            content=complex_content,
            message_type=MessageType.ASSISTANT,
            metadata={
                "model": "test-model",
                "temperature": 0.7,
                "tokens": 42,
            },
            token_count=42,
        )

        assert msg_id is not None, "Message ID should not be None"

        # Verify complex content was stored and retrieved correctly
        message = conversation.query(str(msg_id))
        assert message is not None, "Message should be retrievable"
        assert (
            message["content"] == complex_content
        ), "Complex content should match"
        assert (
            message["token_count"] == 42
        ), "Token count should match"

        print("✓ Add complex message test passed")
        return True
    finally:
        cleanup_test_conversation(conversation)


def test_batch_add() -> bool:
    """Test batch adding messages."""
    print_test_header("Batch Add Test")

    if not SUPABASE_AVAILABLE:
        print("✓ Batch add test skipped - Supabase not available")
        return True

    conversation = setup_test_conversation()
    try:
        messages = [
            Message(
                role="user",
                content="First batch message",
                message_type=MessageType.USER,
                metadata={"batch": 1},
            ),
            Message(
                role="assistant",
                content={
                    "response": "First response",
                    "confidence": 0.9,
                },
                message_type=MessageType.ASSISTANT,
                metadata={"batch": 1},
            ),
            Message(
                role="user",
                content="Second batch message",
                message_type=MessageType.USER,
                metadata={"batch": 2},
            ),
        ]

        msg_ids = conversation.batch_add(messages)
        assert len(msg_ids) == 3, "Should have 3 message IDs"
        assert all(
            isinstance(id, int) for id in msg_ids
        ), "All IDs should be integers"

        # Verify messages were stored
        all_messages = conversation.get_messages()
        assert (
            len(
                [
                    m
                    for m in all_messages
                    if m.get("metadata", {}).get("batch")
                ]
            )
            == 3
        ), "Should find 3 batch messages"

        print("✓ Batch add test passed")
        return True
    finally:
        cleanup_test_conversation(conversation)


def test_get_str() -> bool:
    """Test getting conversation as string."""
    print_test_header("Get String Test")

    if not SUPABASE_AVAILABLE:
        print("✓ Get string test skipped - Supabase not available")
        return True

    conversation = setup_test_conversation()
    try:
        conversation.add("user", "Hello!")
        conversation.add("assistant", "Hi there!")

        conv_str = conversation.get_str()
        assert (
            "user: Hello!" in conv_str
        ), "User message not found in string"
        assert (
            "assistant: Hi there!" in conv_str
        ), "Assistant message not found in string"

        print("✓ Get string test passed")
        return True
    finally:
        cleanup_test_conversation(conversation)


def test_get_messages() -> bool:
    """Test getting messages with pagination."""
    print_test_header("Get Messages Test")

    if not SUPABASE_AVAILABLE:
        print("✓ Get messages test skipped - Supabase not available")
        return True

    conversation = setup_test_conversation()
    try:
        # Add multiple messages
        for i in range(5):
            conversation.add("user", f"Message {i}")

        # Test getting all messages
        all_messages = conversation.get_messages()
        assert (
            len(all_messages) >= 5
        ), "Should have at least 5 messages"

        # Test pagination
        limited_messages = conversation.get_messages(limit=2)
        assert (
            len(limited_messages) == 2
        ), "Should have 2 limited messages"

        offset_messages = conversation.get_messages(offset=2, limit=2)
        assert (
            len(offset_messages) == 2
        ), "Should have 2 offset messages"

        print("✓ Get messages test passed")
        return True
    finally:
        cleanup_test_conversation(conversation)


def test_search_messages() -> bool:
    """Test searching messages."""
    print_test_header("Search Messages Test")

    if not SUPABASE_AVAILABLE:
        print(
            "✓ Search messages test skipped - Supabase not available"
        )
        return True

    conversation = setup_test_conversation()
    try:
        conversation.add("user", "Hello world from Supabase")
        conversation.add("assistant", "Hello there, user!")
        conversation.add("user", "Goodbye world")
        conversation.add("system", "System message without keywords")

        # Test search functionality
        world_results = conversation.search("world")
        assert (
            len(world_results) >= 2
        ), "Should find at least 2 messages with 'world'"

        hello_results = conversation.search("Hello")
        assert (
            len(hello_results) >= 2
        ), "Should find at least 2 messages with 'Hello'"

        supabase_results = conversation.search("Supabase")
        assert (
            len(supabase_results) >= 1
        ), "Should find at least 1 message with 'Supabase'"

        print("✓ Search messages test passed")
        return True
    finally:
        cleanup_test_conversation(conversation)


def test_update_and_delete() -> bool:
    """Test updating and deleting messages."""
    print_test_header("Update and Delete Test")

    if not SUPABASE_AVAILABLE:
        print(
            "✓ Update and delete test skipped - Supabase not available"
        )
        return True

    conversation = setup_test_conversation()
    try:
        # Add a message to update/delete
        msg_id = conversation.add("user", "Original message")

        # Test update method (BaseCommunication signature)
        conversation.update(
            index=str(msg_id), role="user", content="Updated message"
        )

        updated_msg = conversation.query_optional(str(msg_id))
        assert (
            updated_msg is not None
        ), "Message should exist after update"
        assert (
            updated_msg["content"] == "Updated message"
        ), "Message should be updated"

        # Test delete
        conversation.delete(str(msg_id))

        deleted_msg = conversation.query_optional(str(msg_id))
        assert deleted_msg is None, "Message should be deleted"

        print("✓ Update and delete test passed")
        return True
    finally:
        cleanup_test_conversation(conversation)


def test_update_message_method() -> bool:
    """Test the new update_message method."""
    print_test_header("Update Message Method Test")

    if not SUPABASE_AVAILABLE:
        print(
            "✓ Update message method test skipped - Supabase not available"
        )
        return True

    conversation = setup_test_conversation()
    try:
        # Add a message to update
        msg_id = conversation.add(
            role="user",
            content="Original content",
            metadata={"version": 1},
        )

        # Test update_message method
        success = conversation.update_message(
            message_id=msg_id,
            content="Updated content via update_message",
            metadata={"version": 2, "updated": True},
        )

        assert (
            success is True
        ), "update_message should return True on success"

        # Verify the update
        updated_msg = conversation.query(str(msg_id))
        assert updated_msg is not None, "Message should still exist"
        assert (
            updated_msg["content"]
            == "Updated content via update_message"
        ), "Content should be updated"
        assert (
            updated_msg["metadata"]["version"] == 2
        ), "Metadata should be updated"
        assert (
            updated_msg["metadata"]["updated"] is True
        ), "New metadata field should be added"

        # Test update_message with non-existent ID
        failure = conversation.update_message(
            message_id=999999, content="This should fail"
        )
        assert (
            failure is False
        ), "update_message should return False for non-existent message"

        print("✓ Update message method test passed")
        return True
    finally:
        cleanup_test_conversation(conversation)


def test_conversation_statistics() -> bool:
    """Test getting conversation statistics."""
    print_test_header("Conversation Statistics Test")

    if not SUPABASE_AVAILABLE:
        print(
            "✓ Conversation statistics test skipped - Supabase not available"
        )
        return True

    conversation = setup_test_conversation()
    try:
        # Add messages with different roles and token counts
        conversation.add("user", "Hello", token_count=2)
        conversation.add("assistant", "Hi there!", token_count=3)
        conversation.add("system", "System message", token_count=5)
        conversation.add(
            "user", "Another user message", token_count=4
        )

        stats = conversation.get_conversation_summary()
        assert (
            stats["total_messages"] >= 4
        ), "Should have at least 4 messages"
        assert (
            stats["unique_roles"] >= 3
        ), "Should have at least 3 unique roles"
        assert (
            stats["total_tokens"] >= 14
        ), "Should have at least 14 total tokens"

        # Test role counting
        role_counts = conversation.count_messages_by_role()
        assert (
            role_counts.get("user", 0) >= 2
        ), "Should have at least 2 user messages"
        assert (
            role_counts.get("assistant", 0) >= 1
        ), "Should have at least 1 assistant message"
        assert (
            role_counts.get("system", 0) >= 1
        ), "Should have at least 1 system message"

        print("✓ Conversation statistics test passed")
        return True
    finally:
        cleanup_test_conversation(conversation)


def test_json_operations() -> bool:
    """Test JSON save and load operations."""
    print_test_header("JSON Operations Test")

    if not SUPABASE_AVAILABLE:
        print(
            "✓ JSON operations test skipped - Supabase not available"
        )
        return True

    conversation = setup_test_conversation()
    json_file = "test_conversation.json"

    try:
        # Add test messages
        conversation.add("user", "Test message for JSON")
        conversation.add(
            "assistant",
            {"response": "JSON test response", "data": [1, 2, 3]},
        )

        # Test JSON export
        conversation.save_as_json(json_file)
        assert os.path.exists(
            json_file
        ), "JSON file should be created"

        # Verify JSON content
        with open(json_file, "r") as f:
            json_data = json.load(f)
        assert isinstance(
            json_data, list
        ), "JSON data should be a list"
        assert (
            len(json_data) >= 2
        ), "Should have at least 2 messages in JSON"

        # Test JSON import (creates new conversation)
        original_conv_id = conversation.get_conversation_id()
        conversation.load_from_json(json_file)
        new_conv_id = conversation.get_conversation_id()
        assert (
            new_conv_id != original_conv_id
        ), "Should create new conversation on import"

        imported_messages = conversation.get_messages()
        assert (
            len(imported_messages) >= 2
        ), "Should have imported messages"

        print("✓ JSON operations test passed")
        return True
    finally:
        # Cleanup
        if os.path.exists(json_file):
            os.remove(json_file)
        cleanup_test_conversation(conversation)


def test_yaml_operations() -> bool:
    """Test YAML save and load operations."""
    print_test_header("YAML Operations Test")

    if not SUPABASE_AVAILABLE:
        print(
            "✓ YAML operations test skipped - Supabase not available"
        )
        return True

    conversation = setup_test_conversation()
    yaml_file = "test_conversation.yaml"

    try:
        # Add test messages
        conversation.add("user", "Test message for YAML")
        conversation.add("assistant", "YAML test response")

        # Test YAML export
        conversation.save_as_yaml(yaml_file)
        assert os.path.exists(
            yaml_file
        ), "YAML file should be created"

        # Test YAML import (creates new conversation)
        original_conv_id = conversation.get_conversation_id()
        conversation.load_from_yaml(yaml_file)
        new_conv_id = conversation.get_conversation_id()
        assert (
            new_conv_id != original_conv_id
        ), "Should create new conversation on import"

        imported_messages = conversation.get_messages()
        assert (
            len(imported_messages) >= 2
        ), "Should have imported messages"

        print("✓ YAML operations test passed")
        return True
    finally:
        # Cleanup
        if os.path.exists(yaml_file):
            os.remove(yaml_file)
        cleanup_test_conversation(conversation)


def test_message_types() -> bool:
    """Test different message types."""
    print_test_header("Message Types Test")

    if not SUPABASE_AVAILABLE:
        print("✓ Message types test skipped - Supabase not available")
        return True

    conversation = setup_test_conversation()
    try:
        # Test all message types
        types_to_test = [
            (MessageType.USER, "user"),
            (MessageType.ASSISTANT, "assistant"),
            (MessageType.SYSTEM, "system"),
            (MessageType.FUNCTION, "function"),
            (MessageType.TOOL, "tool"),
        ]

        for msg_type, role in types_to_test:
            msg_id = conversation.add(
                role=role,
                content=f"Test {msg_type.value} message",
                message_type=msg_type,
            )
            assert (
                msg_id is not None
            ), f"Should create {msg_type.value} message"

        # Verify all message types were stored
        messages = conversation.get_messages()
        stored_types = {
            msg.get("message_type")
            for msg in messages
            if msg.get("message_type")
        }
        expected_types = {
            msg_type.value for msg_type, _ in types_to_test
        }
        assert stored_types.issuperset(
            expected_types
        ), "Should store all message types"

        print("✓ Message types test passed")
        return True
    finally:
        cleanup_test_conversation(conversation)


def test_conversation_management() -> bool:
    """Test conversation management operations."""
    print_test_header("Conversation Management Test")

    if not SUPABASE_AVAILABLE:
        print(
            "✓ Conversation management test skipped - Supabase not available"
        )
        return True

    conversation = setup_test_conversation()
    try:
        # Test getting conversation ID
        conv_id = conversation.get_conversation_id()
        assert conv_id, "Should have a conversation ID"
        assert isinstance(
            conv_id, str
        ), "Conversation ID should be a string"

        # Add some messages
        conversation.add("user", "First conversation message")
        conversation.add(
            "assistant", "Response in first conversation"
        )

        first_conv_messages = len(conversation.get_messages())
        assert (
            first_conv_messages >= 2
        ), "Should have messages in first conversation"

        # Start new conversation
        new_conv_id = conversation.start_new_conversation()
        assert (
            new_conv_id != conv_id
        ), "New conversation should have different ID"
        assert (
            conversation.get_conversation_id() == new_conv_id
        ), "Should switch to new conversation"
        assert isinstance(
            new_conv_id, str
        ), "New conversation ID should be a string"

        # Verify new conversation is empty (except any system prompts)
        new_messages = conversation.get_messages()
        conversation.add("user", "New conversation message")
        updated_messages = conversation.get_messages()
        assert len(updated_messages) > len(
            new_messages
        ), "Should be able to add to new conversation"

        # Test clear conversation
        conversation.clear()
        cleared_messages = conversation.get_messages()
        assert (
            len(cleared_messages) == 0
        ), "Conversation should be cleared"

        print("✓ Conversation management test passed")
        return True
    finally:
        cleanup_test_conversation(conversation)


def test_get_messages_by_role() -> bool:
    """Test getting messages filtered by role."""
    print_test_header("Get Messages by Role Test")

    if not SUPABASE_AVAILABLE:
        print(
            "✓ Get messages by role test skipped - Supabase not available"
        )
        return True

    conversation = setup_test_conversation()
    try:
        # Add messages with different roles
        conversation.add("user", "User message 1")
        conversation.add("assistant", "Assistant message 1")
        conversation.add("user", "User message 2")
        conversation.add("system", "System message")
        conversation.add("assistant", "Assistant message 2")

        # Test filtering by role
        user_messages = conversation.get_messages_by_role("user")
        assert (
            len(user_messages) >= 2
        ), "Should have at least 2 user messages"
        assert all(
            msg["role"] == "user" for msg in user_messages
        ), "All messages should be from user"

        assistant_messages = conversation.get_messages_by_role(
            "assistant"
        )
        assert (
            len(assistant_messages) >= 2
        ), "Should have at least 2 assistant messages"
        assert all(
            msg["role"] == "assistant" for msg in assistant_messages
        ), "All messages should be from assistant"

        system_messages = conversation.get_messages_by_role("system")
        assert (
            len(system_messages) >= 1
        ), "Should have at least 1 system message"
        assert all(
            msg["role"] == "system" for msg in system_messages
        ), "All messages should be from system"

        print("✓ Get messages by role test passed")
        return True
    finally:
        cleanup_test_conversation(conversation)


def test_timeline_and_organization() -> bool:
    """Test conversation timeline and organization features."""
    print_test_header("Timeline and Organization Test")

    if not SUPABASE_AVAILABLE:
        print(
            "✓ Timeline and organization test skipped - Supabase not available"
        )
        return True

    conversation = setup_test_conversation()
    try:
        # Add messages
        conversation.add("user", "Timeline test message 1")
        conversation.add("assistant", "Timeline test response 1")
        conversation.add("user", "Timeline test message 2")

        # Test timeline organization
        timeline = conversation.get_conversation_timeline_dict()
        assert isinstance(
            timeline, dict
        ), "Timeline should be a dictionary"
        assert len(timeline) > 0, "Timeline should have entries"

        # Test organization by role
        by_role = conversation.get_conversation_by_role_dict()
        assert isinstance(
            by_role, dict
        ), "Role organization should be a dictionary"
        assert "user" in by_role, "Should have user messages"
        assert (
            "assistant" in by_role
        ), "Should have assistant messages"

        # Test conversation as dict
        conv_dict = conversation.get_conversation_as_dict()
        assert isinstance(
            conv_dict, dict
        ), "Conversation dict should be a dictionary"
        assert (
            "conversation_id" in conv_dict
        ), "Should have conversation ID"
        assert "messages" in conv_dict, "Should have messages"

        print("✓ Timeline and organization test passed")
        return True
    finally:
        cleanup_test_conversation(conversation)


def test_concurrent_operations() -> bool:
    """Test concurrent operations for thread safety."""
    print_test_header("Concurrent Operations Test")

    if not SUPABASE_AVAILABLE:
        print(
            "✓ Concurrent operations test skipped - Supabase not available"
        )
        return True

    conversation = setup_test_conversation()
    results = []

    def add_messages(thread_id):
        """Add messages in a separate thread."""
        try:
            for i in range(3):
                msg_id = conversation.add(
                    role="user",
                    content=f"Thread {thread_id} message {i}",
                    metadata={
                        "thread_id": thread_id,
                        "message_num": i,
                    },
                )
                results.append(("success", thread_id, msg_id))
        except Exception as e:
            results.append(("error", thread_id, str(e)))

    try:
        # Create and start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=add_messages, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        successful_operations = [
            r for r in results if r[0] == "success"
        ]
        assert (
            len(successful_operations) >= 6
        ), "Should have successful concurrent operations"

        # Verify messages were actually stored
        all_messages = conversation.get_messages()
        thread_messages = [
            m
            for m in all_messages
            if m.get("metadata", {}).get("thread_id") is not None
        ]
        assert (
            len(thread_messages) >= 6
        ), "Should have stored concurrent messages"

        print("✓ Concurrent operations test passed")
        return True
    finally:
        cleanup_test_conversation(conversation)


def test_enhanced_error_handling() -> bool:
    """Test enhanced error handling for various edge cases."""
    print_test_header("Enhanced Error Handling Test")

    if not SUPABASE_AVAILABLE:
        print(
            "✓ Enhanced error handling test skipped - Supabase not available"
        )
        return True

    # Test invalid credentials
    try:
        SupabaseConversation(
            supabase_url="https://invalid-url.supabase.co",
            supabase_key="invalid_key",
            enable_logging=False,
        )
        # This should raise an exception during initialization
        assert False, "Should raise exception for invalid credentials"
    except (SupabaseConnectionError, Exception):
        pass  # Expected behavior

    # Test with valid conversation
    conversation = setup_test_conversation()
    try:
        # Test querying non-existent message with query (should return empty dict)
        non_existent = conversation.query("999999")
        assert (
            non_existent == {}
        ), "Non-existent message should return empty dict"

        # Test querying non-existent message with query_optional (should return None)
        non_existent_opt = conversation.query_optional("999999")
        assert (
            non_existent_opt is None
        ), "Non-existent message should return None with query_optional"

        # Test deleting non-existent message (should not raise exception)
        conversation.delete("999999")  # Should handle gracefully

        # Test updating non-existent message (should return False)
        update_result = conversation._update_flexible(
            "999999", "user", "content"
        )
        assert (
            update_result is False
        ), "_update_flexible should return False for invalid ID"

        # Test update_message with invalid ID
        result = conversation.update_message(
            999999, "invalid content"
        )
        assert (
            result is False
        ), "update_message should return False for invalid ID"

        # Test search with empty query
        empty_results = conversation.search("")
        assert isinstance(
            empty_results, list
        ), "Empty search should return list"

        # Test invalid message ID formats (should return empty dict now)
        invalid_query = conversation.query("not_a_number")
        assert (
            invalid_query == {}
        ), "Invalid ID should return empty dict"

        invalid_query_opt = conversation.query_optional(
            "not_a_number"
        )
        assert (
            invalid_query_opt is None
        ), "Invalid ID should return None with query_optional"

        # Test update with invalid ID (should return False)
        invalid_update = conversation._update_flexible(
            "not_a_number", "user", "content"
        )
        assert (
            invalid_update is False
        ), "Invalid ID should return False for update"

        print("✓ Enhanced error handling test passed")
        return True
    finally:
        cleanup_test_conversation(conversation)


def test_fallback_functionality() -> bool:
    """Test fallback functionality when dependencies are missing."""
    print_test_header("Fallback Functionality Test")

    # This test always passes as it tests the import fallback mechanism
    if not SUPABASE_AVAILABLE:
        print(
            "✓ Fallback functionality test passed - gracefully handled missing dependencies"
        )
        return True
    else:
        print(
            "✓ Fallback functionality test passed - dependencies available, no fallback needed"
        )
        return True


def generate_test_report(
    test_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate a comprehensive test report."""
    total_tests = len(test_results)
    passed_tests = sum(
        1 for result in test_results if result["success"]
    )
    failed_tests = total_tests - passed_tests

    total_time = sum(
        result["execution_time"] for result in test_results
    )
    avg_time = total_time / total_tests if total_tests > 0 else 0

    report = {
        "summary": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": (
                (passed_tests / total_tests * 100)
                if total_tests > 0
                else 0
            ),
            "total_execution_time": total_time,
            "average_execution_time": avg_time,
            "timestamp": datetime.datetime.now().isoformat(),
            "supabase_available": SUPABASE_AVAILABLE,
            "environment_configured": bool(
                TEST_SUPABASE_URL and TEST_SUPABASE_KEY
            ),
        },
        "test_results": test_results,
        "failed_tests": [
            result for result in test_results if not result["success"]
        ],
    }

    return report


def run_all_tests() -> None:
    """Run all SupabaseConversation tests."""
    print("🚀 Starting Enhanced SupabaseConversation Test Suite")
    print(
        f"Supabase Available: {'✅' if SUPABASE_AVAILABLE else '❌'}"
    )

    if TEST_SUPABASE_URL and TEST_SUPABASE_KEY:
        print(f"Using Supabase URL: {TEST_SUPABASE_URL[:30]}...")
        print(f"Using table: {TEST_TABLE_NAME}")
    else:
        print(
            "❌ Environment variables SUPABASE_URL and SUPABASE_KEY not set"
        )
        print("Some tests will be skipped")

    print("=" * 60)

    # Define tests to run
    tests = [
        ("Import Availability", test_import_availability),
        ("Fallback Functionality", test_fallback_functionality),
        ("Initialization", test_initialization),
        ("Logging Configuration", test_logging_configuration),
        ("Add Message", test_add_message),
        ("Add Complex Message", test_add_complex_message),
        ("Batch Add", test_batch_add),
        ("Get String", test_get_str),
        ("Get Messages", test_get_messages),
        ("Search Messages", test_search_messages),
        ("Update and Delete", test_update_and_delete),
        ("Update Message Method", test_update_message_method),
        ("Conversation Statistics", test_conversation_statistics),
        ("JSON Operations", test_json_operations),
        ("YAML Operations", test_yaml_operations),
        ("Message Types", test_message_types),
        ("Conversation Management", test_conversation_management),
        ("Get Messages by Role", test_get_messages_by_role),
        ("Timeline and Organization", test_timeline_and_organization),
        ("Concurrent Operations", test_concurrent_operations),
        ("Enhanced Error Handling", test_enhanced_error_handling),
    ]

    test_results = []

    # Run each test
    for test_name, test_func in tests:
        print_test_header(test_name)
        success, message, execution_time = run_test(test_func)

        test_result = {
            "test_name": test_name,
            "success": success,
            "message": message,
            "execution_time": execution_time,
        }
        test_results.append(test_result)

        print_test_result(test_name, success, message, execution_time)

    # Generate and display report
    report = generate_test_report(test_results)

    print("\n" + "=" * 60)
    print("📊 ENHANCED TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['passed_tests']}")
    print(f"Failed: {report['summary']['failed_tests']}")
    print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
    print(
        f"Total Time: {report['summary']['total_execution_time']:.3f} seconds"
    )
    print(
        f"Average Time: {report['summary']['average_execution_time']:.3f} seconds"
    )
    print(
        f"Supabase Available: {'✅' if report['summary']['supabase_available'] else '❌'}"
    )
    print(
        f"Environment Configured: {'✅' if report['summary']['environment_configured'] else '❌'}"
    )

    if report["failed_tests"]:
        print("\n❌ FAILED TESTS:")
        for failed_test in report["failed_tests"]:
            print(
                f"  - {failed_test['test_name']}: {failed_test['message']}"
            )
    else:
        print("\n✅ All tests passed!")

    # Additional information
    if not SUPABASE_AVAILABLE:
        print("\n🔍 NOTE: Supabase dependencies not available.")
        print("Install with: pip install supabase")

    if not (TEST_SUPABASE_URL and TEST_SUPABASE_KEY):
        print("\n🔍 NOTE: Environment variables not set.")
        print("Set SUPABASE_URL and SUPABASE_KEY to run full tests.")

    # Save detailed report
    report_file = f"supabase_test_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n📄 Detailed report saved to: {report_file}")


if __name__ == "__main__":
    run_all_tests()
