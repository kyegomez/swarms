"""
Interactive agent to manually test conversation string caching.
Run: python examples/conversation_cache_interactive.py
Type messages, see cache stats update after each response.
"""

from swarms import Agent


def main() -> None:
    agent = Agent(
        agent_name="CacheTestAgent",
        model_name="claude-sonnet-4-5",
        max_loops=1,
        verbose=False,
        temperature=1.0,
    )
    agent.short_memory.caching = True

    print("\n=== Conversation Cache Interactive Test ===")
    print(
        "Type your messages. After each response, cache stats are shown."
    )
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        response = agent.run(user_input)
        print(f"\nAgent: {response}\n")

        stats = agent.short_memory.get_cache_stats()
        total_calls = stats["hits"] + stats["misses"]
        print("--- Cache Stats ---")
        print(f"  Total calls: {total_calls}")
        print(
            f"  Hits:        {stats['hits']}  (returned cached string)"
        )
        print(
            f"  Misses:      {stats['misses']}  (rebuilt the string)"
        )
        print(f"  Hit rate:    {stats['hit_rate']:.0%}")
        print(f"  Tokens in current cache: {stats['cached_tokens']}")
        print("-------------------\n")


if __name__ == "__main__":
    main()
