from swarms.structs.conversation import Conversation

# Example usage
# conversation = Conversation()
conversation = Conversation(token_count=True)
conversation.add("user", "Hello, how are you?")
conversation.add("assistant", "I am doing well, thanks.")
# conversation.add(
#     "assistant", {"name": "tool_1", "output": "Hello, how are you?"}
# )
# print(conversation.return_json())

# # print(conversation.get_last_message_as_string())
print(conversation.return_json())
print(conversation.to_dict())
# # conversation.add("assistant", "I am doing well, thanks.")
# # # print(conversation.to_json())
# print(type(conversation.to_dict()))
# print(conversation.to_yaml())
