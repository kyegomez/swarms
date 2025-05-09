from swarms.structs.conversation import Conversation
import time
import threading
import random
from typing import List


def test_conversation_cache():
    """
    Test the caching functionality of the Conversation class.
    This test demonstrates:
    1. Cache hits and misses
    2. Token counting with caching
    3. Cache statistics
    4. Thread safety
    5. Different content types
    6. Edge cases
    7. Performance metrics
    """
    print("\n=== Testing Conversation Cache ===")

    # Create a conversation with caching enabled
    conv = Conversation(cache_enabled=True)

    # Test 1: Basic caching with repeated messages
    print("\nTest 1: Basic caching with repeated messages")
    message = "This is a test message that should be cached"

    # First add (should be a cache miss)
    print("\nAdding first message...")
    conv.add("user", message)
    time.sleep(0.1)  # Wait for token counting thread

    # Second add (should be a cache hit)
    print("\nAdding same message again...")
    conv.add("user", message)
    time.sleep(0.1)  # Wait for token counting thread

    # Check cache stats
    stats = conv.get_cache_stats()
    print("\nCache stats after repeated message:")
    print(f"Hits: {stats['hits']}")
    print(f"Misses: {stats['misses']}")
    print(f"Cached tokens: {stats['cached_tokens']}")
    print(f"Hit rate: {stats['hit_rate']:.2%}")

    # Test 2: Different content types
    print("\nTest 2: Different content types")

    # Test with dictionary
    dict_content = {"key": "value", "nested": {"inner": "data"}}
    print("\nAdding dictionary content...")
    conv.add("user", dict_content)
    time.sleep(0.1)

    # Test with list
    list_content = ["item1", "item2", {"nested": "data"}]
    print("\nAdding list content...")
    conv.add("user", list_content)
    time.sleep(0.1)

    # Test 3: Thread safety
    print("\nTest 3: Thread safety with concurrent adds")

    def add_message(msg):
        conv.add("user", msg)

    # Add multiple messages concurrently
    messages = [f"Concurrent message {i}" for i in range(5)]
    for msg in messages:
        add_message(msg)

    time.sleep(0.5)  # Wait for all token counting threads

    # Test 4: Cache with different message lengths
    print("\nTest 4: Cache with different message lengths")

    # Short message
    short_msg = "Short"
    conv.add("user", short_msg)
    time.sleep(0.1)

    # Long message
    long_msg = "This is a much longer message that should have more tokens and might be cached differently"
    conv.add("user", long_msg)
    time.sleep(0.1)

    # Test 5: Cache statistics after all tests
    print("\nTest 5: Final cache statistics")
    final_stats = conv.get_cache_stats()
    print("\nFinal cache stats:")
    print(f"Total hits: {final_stats['hits']}")
    print(f"Total misses: {final_stats['misses']}")
    print(f"Total cached tokens: {final_stats['cached_tokens']}")
    print(f"Total tokens: {final_stats['total_tokens']}")
    print(f"Overall hit rate: {final_stats['hit_rate']:.2%}")

    # Test 6: Display conversation with cache status
    print("\nTest 6: Display conversation with cache status")
    print("\nConversation history:")
    print(conv.get_str())

    # Test 7: Cache disabled
    print("\nTest 7: Cache disabled")
    conv_disabled = Conversation(cache_enabled=False)
    conv_disabled.add("user", message)
    time.sleep(0.1)
    conv_disabled.add("user", message)
    time.sleep(0.1)

    disabled_stats = conv_disabled.get_cache_stats()
    print("\nCache stats with caching disabled:")
    print(f"Hits: {disabled_stats['hits']}")
    print(f"Misses: {disabled_stats['misses']}")
    print(f"Cached tokens: {disabled_stats['cached_tokens']}")

    # Test 8: High concurrency stress test
    print("\nTest 8: High concurrency stress test")
    conv_stress = Conversation(cache_enabled=True)

    def stress_test_worker(messages: List[str]):
        for msg in messages:
            conv_stress.add("user", msg)
            time.sleep(random.uniform(0.01, 0.05))

    # Create multiple threads with different messages
    threads = []
    for i in range(5):
        thread_messages = [
            f"Stress test message {i}_{j}" for j in range(10)
        ]
        t = threading.Thread(
            target=stress_test_worker, args=(thread_messages,)
        )
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    time.sleep(0.5)  # Wait for token counting
    stress_stats = conv_stress.get_cache_stats()
    print("\nStress test stats:")
    print(
        f"Total messages: {stress_stats['hits'] + stress_stats['misses']}"
    )
    print(f"Cache hits: {stress_stats['hits']}")
    print(f"Cache misses: {stress_stats['misses']}")

    # Test 9: Complex nested structures
    print("\nTest 9: Complex nested structures")
    complex_content = {
        "nested": {
            "array": [1, 2, 3, {"deep": "value"}],
            "object": {
                "key": "value",
                "nested_array": ["a", "b", "c"],
            },
        },
        "simple": "value",
    }

    # Add complex content multiple times
    for _ in range(3):
        conv.add("user", complex_content)
        time.sleep(0.1)

    # Test 10: Large message test
    print("\nTest 10: Large message test")
    large_message = "x" * 10000  # 10KB message
    conv.add("user", large_message)
    time.sleep(0.1)

    # Test 11: Mixed content types in sequence
    print("\nTest 11: Mixed content types in sequence")
    mixed_sequence = [
        "Simple string",
        {"key": "value"},
        ["array", "items"],
        "Simple string",  # Should be cached
        {"key": "value"},  # Should be cached
        ["array", "items"],  # Should be cached
    ]

    for content in mixed_sequence:
        conv.add("user", content)
        time.sleep(0.1)

    # Test 12: Cache performance metrics
    print("\nTest 12: Cache performance metrics")
    start_time = time.time()

    # Add 100 messages quickly
    for i in range(100):
        conv.add("user", f"Performance test message {i}")

    end_time = time.time()
    performance_stats = conv.get_cache_stats()

    print("\nPerformance metrics:")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Messages per second: {100 / (end_time - start_time):.2f}")
    print(f"Cache hit rate: {performance_stats['hit_rate']:.2%}")

    # Test 13: Cache with special characters
    print("\nTest 13: Cache with special characters")
    special_chars = [
        "Hello! @#$%^&*()",
        "Unicode: 你好世界",
        "Emoji: 😀🎉🌟",
        "Hello! @#$%^&*()",  # Should be cached
        "Unicode: 你好世界",  # Should be cached
        "Emoji: 😀🎉🌟",  # Should be cached
    ]

    for content in special_chars:
        conv.add("user", content)
        time.sleep(0.1)

    # Test 14: Cache with different roles
    print("\nTest 14: Cache with different roles")
    roles = ["user", "assistant", "system", "function"]
    for role in roles:
        conv.add(role, "Same message different role")
        time.sleep(0.1)

    # Final statistics
    print("\n=== Final Cache Statistics ===")
    final_stats = conv.get_cache_stats()
    print(f"Total hits: {final_stats['hits']}")
    print(f"Total misses: {final_stats['misses']}")
    print(f"Total cached tokens: {final_stats['cached_tokens']}")
    print(f"Total tokens: {final_stats['total_tokens']}")
    print(f"Overall hit rate: {final_stats['hit_rate']:.2%}")

    print("\n=== Cache Testing Complete ===")


if __name__ == "__main__":
    test_conversation_cache()
