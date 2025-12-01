"""
Hierarchical Swarm Live Paragraph Streaming Example

This example demonstrates how to use the streaming callback feature
in the HierarchicalSwarm to see live paragraph formation during agent execution.

The streaming callback allows you to:
- Watch paragraphs build in real-time as tokens accumulate
- See the complete text forming word by word
- Track multiple agents working simultaneously
- View completed paragraphs with timestamps
- Monitor the entire generation process live
"""

import time

from swarms.structs.agent import Agent
from swarms.structs.hierarchical_swarm import HierarchicalSwarm


def streaming_callback(agent_name: str, chunk: str, is_final: bool):
    """
    Example streaming callback function that shows live paragraph formation.

    This function is called whenever an agent produces output during streaming.
    It shows tokens accumulating in real-time to form complete paragraphs.

    Args:
        agent_name (str): The name of the agent producing the output
        chunk (str): The chunk of output (empty string if is_final=True)
        is_final (bool): True when the agent has completed its task
    """
    timestamp = time.strftime("%H:%M:%S")

    # Store accumulated text for each agent to track paragraph formation
    if not hasattr(streaming_callback, "agent_buffers"):
        streaming_callback.agent_buffers = {}
        streaming_callback.paragraph_count = {}

    # Initialize buffers for new agents
    if agent_name not in streaming_callback.agent_buffers:
        streaming_callback.agent_buffers[agent_name] = ""
        streaming_callback.paragraph_count[agent_name] = 1
        print(f"\n🎬 [{timestamp}] {agent_name} starting...")
        print("=" * 60)

    if chunk.strip():
        # Split chunk into tokens (words/punctuation)
        tokens = chunk.replace("\n", " \n ").split()

        for token in tokens:
            # Handle paragraph breaks
            if token == "\n":
                if streaming_callback.agent_buffers[
                    agent_name
                ].strip():
                    print(
                        f"\n📄 [{timestamp}] {agent_name} - Paragraph {streaming_callback.paragraph_count[agent_name]} Complete:"
                    )
                    print(
                        f"{streaming_callback.agent_buffers[agent_name].strip()}"
                    )
                    print("=" * 60)
                    streaming_callback.paragraph_count[
                        agent_name
                    ] += 1
                    streaming_callback.agent_buffers[agent_name] = ""
            else:
                # Add token to buffer and show live accumulation
                streaming_callback.agent_buffers[agent_name] += (
                    token + " "
                )

                # Clear line and show current paragraph
                print(
                    f"\r[{timestamp}] {agent_name} | {streaming_callback.agent_buffers[agent_name].strip()}",
                    end="",
                    flush=True,
                )

    if is_final:
        print()  # New line after live updates
        # Print any remaining content as final paragraph
        if streaming_callback.agent_buffers[agent_name].strip():
            print(
                f"\n✅ [{timestamp}] {agent_name} COMPLETED - Final Paragraph:"
            )
            print(
                f"{streaming_callback.agent_buffers[agent_name].strip()}"
            )
            print()

        print(f"🎯 [{timestamp}] {agent_name} finished processing")
        print(
            f"📊 Total paragraphs processed: {streaming_callback.paragraph_count[agent_name] - 1}"
        )
        print("=" * 60)


def create_sample_agents():
    """Create sample agents for the hierarchical swarm."""
    # Marketing Strategist Agent
    marketing_agent = Agent(
        agent_name="MarketingStrategist",
        agent_description="Expert in marketing strategy and campaign planning",
        system_prompt="You are a marketing strategist. Provide creative and effective marketing strategies.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    # Content Creator Agent
    content_agent = Agent(
        agent_name="ContentCreator",
        agent_description="Expert in creating engaging content",
        system_prompt="You are a content creator. Create engaging, well-written content for various platforms.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    # Data Analyst Agent
    analyst_agent = Agent(
        agent_name="DataAnalyst",
        agent_description="Expert in data analysis and insights",
        system_prompt="You are a data analyst. Provide detailed analysis and insights from data.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    return [marketing_agent, content_agent, analyst_agent]


def main():
    """Main function demonstrating hierarchical swarm with streaming."""
    print("🚀 Hierarchical Swarm Streaming Example")
    print("=" * 60)

    # Create agents
    agents = create_sample_agents()

    # Create hierarchical swarm
    swarm = HierarchicalSwarm(
        name="MarketingCampaignSwarm",
        description="A swarm for planning and executing marketing campaigns",
        agents=agents,
        director_model_name="gpt-4o-mini",
        max_loops=1,
        verbose=True,
    )

    # Define the task
    task = """
    Plan and execute a comprehensive marketing campaign for a new tech startup called 'CodeFlow'
    that develops AI-powered code generation tools. The campaign should include:

    1. Target audience analysis
    2. Content strategy development
    3. Social media campaign plan
    4. Performance metrics and KPIs

    Create a detailed campaign plan with specific tactics, timelines, and budget considerations.
    """

    print(f"📋 Task: {task.strip()}")
    print(
        "\n🎯 Starting hierarchical swarm with live paragraph streaming..."
    )
    print("Watch as agents build complete paragraphs in real-time!\n")
    print(
        "Each token accumulates to form readable text, showing the full paragraph as it builds.\n"
    )

    # Run the swarm with streaming callback
    try:
        result = swarm.run(
            task=task, streaming_callback=streaming_callback
        )

        print("\n🎉 Swarm execution completed!")
        print("\n📊 Final Results:")
        print("-" * 30)
        print(result)

    except Exception as e:
        print(f"❌ Error running swarm: {str(e)}")


def simple_callback_example():
    """Simpler example with token-by-token streaming."""
    print("\n🔧 Simple Token-by-Token Callback Example")
    print("=" * 50)

    def simple_callback(agent_name: str, chunk: str, is_final: bool):
        """Simple callback that shows live paragraph formation."""
        if not hasattr(simple_callback, "buffer"):
            simple_callback.buffer = {}
            simple_callback.token_count = {}

        if agent_name not in simple_callback.buffer:
            simple_callback.buffer[agent_name] = ""
            simple_callback.token_count[agent_name] = 0

        if chunk.strip():
            tokens = chunk.replace("\n", " \n ").split()
            for token in tokens:
                if token.strip():
                    simple_callback.token_count[agent_name] += 1
                    simple_callback.buffer[agent_name] += token + " "
                    # Show live accumulation
                    print(
                        f"\r{agent_name} | {simple_callback.buffer[agent_name].strip()}",
                        end="",
                        flush=True,
                    )

        if is_final:
            print()  # New line after live updates
            print(
                f"✓ {agent_name} finished! Total tokens: {simple_callback.token_count[agent_name]}"
            )
            print(
                f"Final text: {simple_callback.buffer[agent_name].strip()}"
            )
            print("-" * 40)

    # Create simple agents
    agents = [
        Agent(
            agent_name="Researcher",
            agent_description="Research specialist",
            system_prompt="You are a researcher. Provide thorough research on given topics.",
            model_name="gpt-4o-mini",
            max_loops=1,
        ),
        Agent(
            agent_name="Writer",
            agent_description="Content writer",
            system_prompt="You are a writer. Create clear, concise content.",
            model_name="gpt-4o-mini",
            max_loops=1,
        ),
    ]

    swarm = HierarchicalSwarm(
        name="SimpleSwarm",
        description="Simple swarm example",
        agents=agents,
        director_model_name="gpt-4o-mini",
        max_loops=1,
    )

    task = "Research the benefits of renewable energy and write a summary article."

    print(f"Task: {task}")
    result = swarm.run(task=task, streaming_callback=simple_callback)
    print(f"\nResult: {result}")


if __name__ == "__main__":
    # Run the main streaming example
    main()

    # Uncomment to run the simple example
    # simple_callback_example()
