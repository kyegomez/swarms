import os
import json
import time
import datetime
import yaml
from swarms.structs.conversation import (
    Conversation,
    generate_conversation_id,
)


def run_all_tests():
    """Run all tests for the Conversation class"""
    test_results = []

    def run_test(test_func):
        try:
            test_func()
            test_results.append(f"✅ {test_func.__name__} passed")
        except Exception as e:
            test_results.append(
                f"❌ {test_func.__name__} failed: {str(e)}"
            )

    def test_basic_initialization():
        """Test basic initialization of Conversation"""
        conv = Conversation()
        assert conv.id is not None
        assert conv.conversation_history is not None
        assert isinstance(conv.conversation_history, list)

        # Test with custom ID
        custom_id = generate_conversation_id()
        conv_with_id = Conversation(id=custom_id)
        assert conv_with_id.id == custom_id

        # Test with custom name
        conv_with_name = Conversation(name="Test Conversation")
        assert conv_with_name.name == "Test Conversation"

    def test_initialization_with_settings():
        """Test initialization with various settings"""
        conv = Conversation(
            system_prompt="Test system prompt",
            time_enabled=True,
            autosave=True,
            token_count=True,
            provider="in-memory",
            context_length=4096,
            rules="Test rules",
            custom_rules_prompt="Custom rules",
            user="TestUser:",
            save_as_yaml=True,
            save_as_json_bool=True,
        )

        # Test all settings
        assert conv.system_prompt == "Test system prompt"
        assert conv.time_enabled is True
        assert conv.autosave is True
        assert conv.token_count is True
        assert conv.provider == "in-memory"
        assert conv.context_length == 4096
        assert conv.rules == "Test rules"
        assert conv.custom_rules_prompt == "Custom rules"
        assert conv.user == "TestUser:"
        assert conv.save_as_yaml is True
        assert conv.save_as_json_bool is True

    def test_message_manipulation():
        """Test adding, deleting, and updating messages"""
        conv = Conversation()

        # Test adding messages with different content types
        conv.add("user", "Hello")  # String content
        conv.add("assistant", {"response": "Hi"})  # Dict content
        conv.add("system", ["Hello", "Hi"])  # List content

        assert len(conv.conversation_history) == 3
        assert isinstance(
            conv.conversation_history[1]["content"], dict
        )
        assert isinstance(
            conv.conversation_history[2]["content"], list
        )

        # Test adding multiple messages
        conv.add_multiple(
            ["user", "assistant", "system"],
            ["Hi", "Hello there", "System message"],
        )
        assert len(conv.conversation_history) == 6

        # Test updating message with different content type
        conv.update(0, "user", {"updated": "content"})
        assert isinstance(
            conv.conversation_history[0]["content"], dict
        )

        # Test deleting multiple messages
        conv.delete(0)
        conv.delete(0)
        assert len(conv.conversation_history) == 4

    def test_message_retrieval():
        """Test message retrieval methods"""
        conv = Conversation()

        # Add messages in specific order for testing
        conv.add("user", "Test message")
        conv.add("assistant", "Test response")
        conv.add("system", "System message")

        # Test query - note: messages might have system prompt prepended
        message = conv.query(0)
        assert "Test message" in message["content"]

        # Test search with multiple results
        results = conv.search("Test")
        assert (
            len(results) >= 2
        )  # At least two messages should contain "Test"
        assert any(
            "Test message" in str(msg["content"]) for msg in results
        )
        assert any(
            "Test response" in str(msg["content"]) for msg in results
        )

        # Test get_last_message_as_string
        last_message = conv.get_last_message_as_string()
        assert "System message" in last_message

        # Test return_messages_as_list
        messages_list = conv.return_messages_as_list()
        assert (
            len(messages_list) >= 3
        )  # At least our 3 added messages
        assert any("Test message" in msg for msg in messages_list)

        # Test return_messages_as_dictionary
        messages_dict = conv.return_messages_as_dictionary()
        assert (
            len(messages_dict) >= 3
        )  # At least our 3 added messages
        assert all(isinstance(m, dict) for m in messages_dict)
        assert all(
            {"role", "content"} <= set(m.keys())
            for m in messages_dict
        )

        # Test get_final_message and content
        assert "System message" in conv.get_final_message()
        assert "System message" in conv.get_final_message_content()

        # Test return_all_except_first
        remaining_messages = conv.return_all_except_first()
        assert (
            len(remaining_messages) >= 2
        )  # At least 2 messages after removing first

        # Test return_all_except_first_string
        remaining_string = conv.return_all_except_first_string()
        assert isinstance(remaining_string, str)

    def test_saving_loading():
        """Test saving and loading conversation"""
        # Test with save_enabled
        conv = Conversation(
            save_enabled=True,
            conversations_dir="./test_conversations",
        )
        conv.add("user", "Test save message")

        # Test save_as_json
        test_file = os.path.join(
            "./test_conversations", "test_conversation.json"
        )
        conv.save_as_json(test_file)
        assert os.path.exists(test_file)

        # Test load_from_json
        new_conv = Conversation()
        new_conv.load_from_json(test_file)
        assert len(new_conv.conversation_history) == 1
        assert (
            new_conv.conversation_history[0]["content"]
            == "Test save message"
        )

        # Test class method load_conversation
        loaded_conv = Conversation.load_conversation(
            name=conv.id, conversations_dir="./test_conversations"
        )
        assert loaded_conv.id == conv.id

        # Cleanup
        os.remove(test_file)
        os.rmdir("./test_conversations")

    def test_output_formats():
        """Test different output formats"""
        conv = Conversation()
        conv.add("user", "Test message")
        conv.add("assistant", {"response": "Test"})

        # Test JSON output
        json_output = conv.to_json()
        assert isinstance(json_output, str)
        parsed_json = json.loads(json_output)
        assert len(parsed_json) == 2

        # Test dict output
        dict_output = conv.to_dict()
        assert isinstance(dict_output, list)
        assert len(dict_output) == 2

        # Test YAML output
        yaml_output = conv.to_yaml()
        assert isinstance(yaml_output, str)
        parsed_yaml = yaml.safe_load(yaml_output)
        assert len(parsed_yaml) == 2

        # Test return_json
        json_str = conv.return_json()
        assert isinstance(json_str, str)
        assert len(json.loads(json_str)) == 2

    def test_memory_management():
        """Test memory management functions"""
        conv = Conversation()

        # Test clear
        conv.add("user", "Test message")
        conv.clear()
        assert len(conv.conversation_history) == 0

        # Test clear_memory
        conv.add("user", "Test message")
        conv.clear_memory()
        assert len(conv.conversation_history) == 0

        # Test batch operations
        messages = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
        ]
        conv.batch_add(messages)
        assert len(conv.conversation_history) == 2

        # Test truncate_memory_with_tokenizer
        if conv.tokenizer:  # Only if tokenizer is available
            conv.truncate_memory_with_tokenizer()
            assert len(conv.conversation_history) > 0

    def test_conversation_metadata():
        """Test conversation metadata and listing"""
        test_dir = "./test_conversations_metadata"
        os.makedirs(test_dir, exist_ok=True)

        try:
            # Create a conversation with metadata
            conv = Conversation(
                name="Test Conv",
                system_prompt="System",
                rules="Rules",
                custom_rules_prompt="Custom",
                conversations_dir=test_dir,
                save_enabled=True,
                autosave=True,
            )

            # Add a message to trigger save
            conv.add("user", "Test message")

            # Give a small delay for autosave
            time.sleep(0.1)

            # List conversations and verify
            conversations = Conversation.list_conversations(test_dir)
            assert len(conversations) >= 1
            found_conv = next(
                (
                    c
                    for c in conversations
                    if c["name"] == "Test Conv"
                ),
                None,
            )
            assert found_conv is not None
            assert found_conv["id"] == conv.id

        finally:
            # Cleanup
            import shutil

            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)

    def test_time_enabled_messages():
        """Test time-enabled messages"""
        conv = Conversation(time_enabled=True)
        conv.add("user", "Time test")

        # Verify timestamp in message
        message = conv.conversation_history[0]
        assert "timestamp" in message
        assert isinstance(message["timestamp"], str)

        # Verify time in content when time_enabled is True
        assert "Time:" in message["content"]

    def test_provider_specific():
        """Test provider-specific functionality"""
        # Test in-memory provider
        conv_memory = Conversation(provider="in-memory")
        conv_memory.add("user", "Test")
        assert len(conv_memory.conversation_history) == 1

        # Test mem0 provider if available
        try:
            conv_mem0 = Conversation(provider="mem0")
            conv_mem0.add("user", "Test")
            # Add appropriate assertions based on mem0 behavior
        except Exception:
            pass  # Skip if mem0 is not available

    def test_tool_output():
        """Test tool output handling"""
        conv = Conversation()
        tool_output = {
            "tool_name": "test_tool",
            "output": "test result",
        }
        conv.add_tool_output_to_agent("tool", tool_output)

        assert len(conv.conversation_history) == 1
        assert conv.conversation_history[0]["role"] == "tool"
        assert conv.conversation_history[0]["content"] == tool_output

    def test_autosave_functionality():
        """Test autosave functionality and related features"""
        test_dir = "./test_conversations_autosave"
        os.makedirs(test_dir, exist_ok=True)

        try:
            # Test with autosave and save_enabled True
            conv = Conversation(
                autosave=True,
                save_enabled=True,
                conversations_dir=test_dir,
                name="autosave_test",
            )

            # Add a message and verify it was auto-saved
            conv.add("user", "Test autosave message")
            save_path = os.path.join(test_dir, f"{conv.id}.json")

            # Give a small delay for autosave to complete
            time.sleep(0.1)

            assert os.path.exists(
                save_path
            ), f"Save file not found at {save_path}"

            # Load the saved conversation and verify content
            loaded_conv = Conversation.load_conversation(
                name=conv.id, conversations_dir=test_dir
            )
            found_message = False
            for msg in loaded_conv.conversation_history:
                if "Test autosave message" in str(msg["content"]):
                    found_message = True
                    break
            assert (
                found_message
            ), "Message not found in loaded conversation"

            # Clean up first conversation files
            if os.path.exists(save_path):
                os.remove(save_path)

            # Test with save_enabled=False
            conv_no_save = Conversation(
                autosave=False,  # Changed to False to prevent autosave
                save_enabled=False,
                conversations_dir=test_dir,
            )
            conv_no_save.add("user", "This shouldn't be saved")
            save_path_no_save = os.path.join(
                test_dir, f"{conv_no_save.id}.json"
            )
            time.sleep(0.1)  # Give time for potential save
            assert not os.path.exists(
                save_path_no_save
            ), "File should not exist when save_enabled is False"

        finally:
            # Cleanup
            import shutil

            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)

    def test_advanced_message_handling():
        """Test advanced message handling features"""
        conv = Conversation()

        # Test adding messages with metadata
        metadata = {"timestamp": "2024-01-01", "session_id": "123"}
        conv.add("user", "Test with metadata", metadata=metadata)

        # Test batch operations with different content types
        messages = [
            {"role": "user", "content": "Message 1"},
            {
                "role": "assistant",
                "content": {"response": "Complex response"},
            },
            {"role": "system", "content": ["Multiple", "Items"]},
        ]
        conv.batch_add(messages)
        assert (
            len(conv.conversation_history) == 4
        )  # Including the first message

        # Test message format consistency
        for msg in conv.conversation_history:
            assert "role" in msg
            assert "content" in msg
            if "timestamp" in msg:
                assert isinstance(msg["timestamp"], str)

    def test_conversation_metadata_handling():
        """Test handling of conversation metadata and attributes"""
        test_dir = "./test_conversations_metadata_handling"
        os.makedirs(test_dir, exist_ok=True)

        try:
            # Test initialization with all optional parameters
            conv = Conversation(
                name="Test Conv",
                system_prompt="System Prompt",
                time_enabled=True,
                context_length=2048,
                rules="Test Rules",
                custom_rules_prompt="Custom Rules",
                user="CustomUser:",
                provider="in-memory",
                conversations_dir=test_dir,
                save_enabled=True,
            )

            # Verify all attributes are set correctly
            assert conv.name == "Test Conv"
            assert conv.system_prompt == "System Prompt"
            assert conv.time_enabled is True
            assert conv.context_length == 2048
            assert conv.rules == "Test Rules"
            assert conv.custom_rules_prompt == "Custom Rules"
            assert conv.user == "CustomUser:"
            assert conv.provider == "in-memory"

            # Test saving and loading preserves metadata
            conv.save_as_json()

            # Load using load_conversation
            loaded_conv = Conversation.load_conversation(
                name=conv.id, conversations_dir=test_dir
            )

            # Verify metadata was preserved
            assert loaded_conv.name == "Test Conv"
            assert loaded_conv.system_prompt == "System Prompt"
            assert loaded_conv.rules == "Test Rules"

        finally:
            # Cleanup
            import shutil

            shutil.rmtree(test_dir)

    def test_time_enabled_features():
        """Test time-enabled message features"""
        conv = Conversation(time_enabled=True)

        # Add message and verify timestamp
        conv.add("user", "Time test message")
        message = conv.conversation_history[0]

        # Verify timestamp format
        assert "timestamp" in message
        try:
            datetime.datetime.fromisoformat(message["timestamp"])
        except ValueError:
            assert False, "Invalid timestamp format"

        # Verify time in content
        assert "Time:" in message["content"]
        assert (
            datetime.datetime.now().strftime("%Y-%m-%d")
            in message["content"]
        )

    def test_provider_specific_features():
        """Test provider-specific features and behaviors"""
        # Test in-memory provider
        conv_memory = Conversation(provider="in-memory")
        conv_memory.add("user", "Test in-memory")
        assert len(conv_memory.conversation_history) == 1
        assert (
            "Test in-memory"
            in conv_memory.get_last_message_as_string()
        )

        # Test mem0 provider if available
        try:
            from mem0 import AsyncMemory  # noqa: F401

            # Skip actual mem0 testing since it requires async
            pass
        except ImportError:
            pass

        # Test invalid provider
        invalid_provider = "invalid_provider"
        try:
            Conversation(provider=invalid_provider)
            # If we get here, the provider was accepted when it shouldn't have been
            raise AssertionError(
                f"Should have raised ValueError for provider '{invalid_provider}'"
            )
        except ValueError:
            # This is the expected behavior
            pass

    # Run all tests
    tests = [
        test_basic_initialization,
        test_initialization_with_settings,
        test_message_manipulation,
        test_message_retrieval,
        test_saving_loading,
        test_output_formats,
        test_memory_management,
        test_conversation_metadata,
        test_time_enabled_messages,
        test_provider_specific,
        test_tool_output,
        test_autosave_functionality,
        test_advanced_message_handling,
        test_conversation_metadata_handling,
        test_time_enabled_features,
        test_provider_specific_features,
    ]

    for test in tests:
        run_test(test)

    # Print results
    print("\nTest Results:")
    for result in test_results:
        print(result)


if __name__ == "__main__":
    run_all_tests()
