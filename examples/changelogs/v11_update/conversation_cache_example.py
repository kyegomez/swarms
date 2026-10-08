from swarms.structs.conversation import Conversation

# Opt-in to caching explicitly
conv = Conversation(cache_enabled=True)
conv.add("user", "What is the capital of France?")

# Caching happens inside return_history_as_string()
history = conv.return_history_as_string()
print(history)

stats = conv.get_cache_stats()
# Returns: {"hits": int, "misses": int}  (total_tokens removed)
print(stats)
