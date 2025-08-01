from swarms.structs.conversation import Conversation
from dotenv import load_dotenv
from swarms.utils.litellm_tokenizer import count_tokens

# Load environment variables from .env file
load_dotenv()

def demonstrate_truncation():
    # Using a smaller context length to clearly see the truncation effect
    context_length = 25
    print(f"Creating a conversation instance with context length {context_length}")
    
    # Using Claude model as the tokenizer model
    conversation = Conversation(
        context_length=context_length,
        tokenizer_model_name="claude-3-7-sonnet-20250219"
    )
    
    # Adding first message - short message
    short_message = "Hello, I am a user."
    print(f"\nAdding short message: '{short_message}'")
    conversation.add("user", short_message)
    
    # Display token count
    
    tokens = count_tokens(short_message, conversation.tokenizer_model_name)
    print(f"Short message token count: {tokens}")
    
    # Adding second message - long message, should be truncated
    long_message = "I have a question about artificial intelligence. I want to understand how large language models handle long texts, especially under token constraints. This issue is important because it relates to the model's practicality and effectiveness. I hope to get a detailed answer that helps me understand this complex technical problem."
    print(f"\nAdding long message:\n'{long_message}'")
    conversation.add("assistant", long_message)
    
    # Display long message token count
    tokens = count_tokens(long_message, conversation.tokenizer_model_name)
    print(f"Long message token count: {tokens}")
    
    # Display current conversation total token count
    total_tokens = sum(count_tokens(msg["content"], conversation.tokenizer_model_name) 
                      for msg in conversation.conversation_history)
    print(f"Total token count before truncation: {total_tokens}")
    
    # Print the complete conversation history before truncation
    print("\nConversation history before truncation:")
    for i, msg in enumerate(conversation.conversation_history):
        print(f"[{i}] {msg['role']}: {msg['content']}")
        print(f"    Token count: {count_tokens(msg['content'], conversation.tokenizer_model_name)}")
    
    # Execute truncation
    print("\nExecuting truncation...")
    conversation.truncate_memory_with_tokenizer()
    
    # Print conversation history after truncation
    print("\nConversation history after truncation:")
    for i, msg in enumerate(conversation.conversation_history):
        print(f"[{i}] {msg['role']}: {msg['content']}")
        print(f"    Token count: {count_tokens(msg['content'], conversation.tokenizer_model_name)}")
    
    # Display total token count after truncation
    total_tokens = sum(count_tokens(msg["content"], conversation.tokenizer_model_name) 
                      for msg in conversation.conversation_history)
    print(f"\nTotal token count after truncation: {total_tokens}")
    print(f"Context length limit: {context_length}")
    
    # Verify if successfully truncated below the limit
    if total_tokens <= context_length:
        print("✅ Success: Total token count is now less than or equal to context length limit")
    else:
        print("❌ Failure: Total token count still exceeds context length limit")
    
    # Test sentence boundary truncation
    print("\n\nTesting sentence boundary truncation:")
    sentence_test = Conversation(context_length=15, tokenizer_model_name="claude-3-opus-20240229")
    test_text = "This is the first sentence. This is the second very long sentence that contains a lot of content. This is the third sentence."
    print(f"Original text: '{test_text}'")
    print(f"Original token count: {count_tokens(test_text, sentence_test.tokenizer_model_name)}")
    
    # Using binary search for truncation
    truncated = sentence_test._binary_search_truncate(test_text, 10, sentence_test.tokenizer_model_name)
    print(f"Truncated text: '{truncated}'")
    print(f"Truncated token count: {count_tokens(truncated, sentence_test.tokenizer_model_name)}")
    
    # Check if truncated at period
    if truncated.endswith("."):
        print("✅ Success: Text was truncated at sentence boundary")
    else:
        print("Note: Text was not truncated at sentence boundary")


if __name__ == "__main__":
    demonstrate_truncation()