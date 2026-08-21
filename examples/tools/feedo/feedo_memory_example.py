import os
import sys
from swarms import Agent

# Ensure we can import feedo_tools from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feedo_tools import FeedoMemoryTools

def main():
    # 1. Initialize Feedo Tools 
    # (Requires FEEDO_USAGE_KEY environment variable. You can get a free testnet key at feedo.ink)
    try:
        feedo_tools = FeedoMemoryTools()
    except Exception as e:
        print(f"Failed to initialize Feedo Memory: {e}")
        print("Please set your FEEDO_USAGE_KEY environment variable.")
        return

    # 2. Initialize Swarms Agent with Feedo tools
    agent = Agent(
        agent_name="FeedoResearcher",
        model_name="gpt-4o",
        system_prompt=(
            "You are a research agent. Always use the search_memory tool to check "
            "past context before answering. If you learn something new, use add_memory "
            "to save it. If memory is outdated, use update_memory or delete_memory."
        ),
        tools=feedo_tools.get_tools(),
        max_loops=2,
        autosave=True,
        dashboard=False,
    )

    # 3. Run a task
    print("Running Agent...")
    response = agent.run(
        "Find everything we previously researched about decentralized AI networks, "
        "and update that information with the latest trends in the industry."
    )
    
    print("\nAgent Output:")
    print(response)

if __name__ == "__main__":
    main()
