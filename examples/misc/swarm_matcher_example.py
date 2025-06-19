from swarms.structs.swarm_matcher import (
    SwarmMatcher,
    SwarmMatcherConfig,
)
from dotenv import load_dotenv

load_dotenv()

# Example usage
if __name__ == "__main__":
    # Create configuration
    config = SwarmMatcherConfig(
        backend="openai",  # Using local embeddings for this example
        similarity_threshold=0.6,  # Increase threshold for more strict matching
        cache_embeddings=True,
    )

    # Initialize matcher
    matcher = SwarmMatcher(config)

    task = "I need to build a hierarchical swarm of agents to solve a problem"

    print(matcher.auto_select_swarm(task))
