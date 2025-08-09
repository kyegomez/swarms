from swarms.structs.conversation import Conversation

# Create a conversation object
conversation = Conversation(backend="in-memory")

# Add a message to the conversation
conversation.add(
    role="user", content="Hello, how are you?", category="input"
)

# Add a message to the conversation
conversation.add(
    role="assistant",
    content="I'm good, thank you!",
    category="output",
)

print(
    conversation.export_and_count_categories(
        tokenizer_model_name="claude-3-5-sonnet-20240620"
    )
)
