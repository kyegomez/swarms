from swarms.structs.conversation import (
    Conversation,
    get_conversation_dir,
)
import os
import shutil


def cleanup_test_conversations():
    """Clean up test conversation files after running the example."""
    conv_dir = get_conversation_dir()
    if os.path.exists(conv_dir):
        shutil.rmtree(conv_dir)
        print(
            f"\nCleaned up test conversations directory: {conv_dir}"
        )


def main():
    # Example 1: In-memory only conversation (no saving)
    print("\nExample 1: In-memory conversation (no saving)")
    conv_memory = Conversation(
        name="memory_only_chat",
        autosave=False,
    )
    conv_memory.add("user", "This conversation won't be saved!")
    print(conv_memory.return_history_as_string())

    # Example 2: Conversation with autosaving
    print("\nExample 2: Conversation with autosaving")
    conversation_dir = get_conversation_dir()
    print(f"Conversations will be stored in: {conversation_dir}")

    conv_autosave = Conversation(
        name="autosave_chat",
        conversations_dir=conversation_dir,
        autosave=True,  # Enable autosaving
    )
    print(f"Created new conversation with ID: {conv_autosave.id}")
    print(
        f"This conversation is saved at: {conv_autosave.save_filepath}"
    )

    # Add some messages (each will be autosaved)
    conv_autosave.add("user", "Hello! How are you?")
    conv_autosave.add(
        "assistant",
        "I'm doing well, thank you! How can I help you today?",
    )

    # Example 3: Load from specific file
    print("\nExample 3: Load from specific file")
    custom_file = os.path.join(conversation_dir, "custom_chat.json")

    # Create a conversation and save it to a custom file
    conv_custom = Conversation(
        name="custom_chat",
        save_filepath=custom_file,
    )
    conv_custom.add("user", "This is a custom saved conversation")
    conv_custom.add(
        "assistant", "I'll be saved in a custom location!"
    )
    conv_custom.save_as_json()

    # Now load it specifically
    loaded_conv = Conversation.load_conversation(
        name="custom_chat", load_filepath=custom_file
    )
    print("Loaded custom conversation:")
    print(loaded_conv.return_history_as_string())

    # List all saved conversations
    print("\nAll saved conversations:")
    conversations = Conversation.list_cached_conversations(
        conversation_dir
    )
    for name in conversations:
        print(f"- {name}")


main()
