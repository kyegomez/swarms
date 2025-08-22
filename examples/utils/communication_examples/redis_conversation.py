from swarms.communication.redis_wrap import RedisConversation
import json
import time


def print_messages(conv):
    messages = conv.to_dict()
    print(f"Messages for conversation '{conv.get_name()}':")
    print(json.dumps(messages, indent=4))


# First session - Add messages
print("\n=== First Session ===")
conv = RedisConversation(
    use_embedded_redis=True,
    redis_port=6380,
    token_count=False,
    cache_enabled=False,
    auto_persist=True,
    redis_data_dir="/Users/swarms_wd/.swarms/redis",
    name="my_test_chat",  # Use a friendly name instead of conversation_id
)

# Add messages
conv.add("user", "Hello!")
conv.add("assistant", "Hi there! How can I help?")
conv.add("user", "What's the weather like?")

# Print current messages
print_messages(conv)

# Close the first connection
del conv
time.sleep(2)  # Give Redis time to save

# Second session - Verify persistence
print("\n=== Second Session ===")
conv2 = RedisConversation(
    use_embedded_redis=True,
    redis_port=6380,
    token_count=False,
    cache_enabled=False,
    auto_persist=True,
    redis_data_dir="/Users/swarms_wd/.swarms/redis",
    name="my_test_chat",  # Use the same name to restore the conversation
)

# Print messages from second session
print_messages(conv2)

# You can also change the name if needed
# conv2.set_name("weather_chat")
