import time
from typing import Callable
from swarms.structs.hiearchical_swarm import HierarchicalSwarm
from swarms import Agent


def create_streaming_callback() -> Callable[[str, str, bool], None]:
    """Create a streaming callback that shows live paragraph formation."""

    # Store accumulated text for each agent to track paragraph formation
    agent_buffers = {}
    paragraph_count = {}

    def streaming_callback(agent_name: str, chunk: str, is_final: bool):
        timestamp = time.strftime("%H:%M:%S")

        # Initialize buffers for new agents
        if agent_name not in agent_buffers:
            agent_buffers[agent_name] = ""
            paragraph_count[agent_name] = 1
            print(f"\n🎬 [{timestamp}] {agent_name} starting...")
            print("="*60)

        if chunk.strip():
            # Split chunk into tokens (words/punctuation)
            tokens = chunk.replace('\n', ' \n ').split()

            for token in tokens:
                # Handle paragraph breaks
                if token == '\n':
                    if agent_buffers[agent_name].strip():
                        print(f"\n📄 [{timestamp}] {agent_name} - Paragraph {paragraph_count[agent_name]} Complete:")
                        print(f"{agent_buffers[agent_name].strip()}")
                        print("="*60)
                        paragraph_count[agent_name] += 1
                        agent_buffers[agent_name] = ""
                else:
                    # Add token to buffer and show live accumulation
                    agent_buffers[agent_name] += token + " "

                    # Clear line and show current paragraph
                    print(f"\r[{timestamp}] {agent_name} | {agent_buffers[agent_name].strip()}", end="", flush=True)

        if is_final:
            print()  # New line after live updates
            # Print any remaining content as final paragraph
            if agent_buffers[agent_name].strip():
                print(f"\n✅ [{timestamp}] {agent_name} COMPLETED - Final Paragraph:")
                print(f"{agent_buffers[agent_name].strip()}")
                print()

            print(f"🎯 [{timestamp}] {agent_name} finished processing")
            print(f"📊 Total paragraphs processed: {paragraph_count[agent_name] - 1}")
            print("="*60)

    return streaming_callback


def create_agents():
    """Create specialized agents for the swarm."""
    return [
        Agent(
            agent_name="Research_Agent",
            agent_description="Specialized in gathering and analyzing information",
            system_prompt="You are a research specialist. Provide detailed, accurate information on any topic.",
            model_name="gpt-4o-mini",
            max_loops=1,
            streaming_on=True,
        ),
        Agent(
            agent_name="Analysis_Agent",
            agent_description="Expert at analyzing data and drawing insights",
            system_prompt="You are an analysis expert. Break down complex information and provide clear insights.",
            model_name="gpt-4o-mini",
            max_loops=1,
            streaming_on=True,
        ),
        Agent(
            agent_name="Summary_Agent",
            agent_description="Skilled at creating concise summaries",
            system_prompt="You are a summarization expert. Create clear, concise summaries of complex topics.",
            model_name="gpt-4o-mini",
            max_loops=1,
            streaming_on=True,
        ),
    ]


if __name__ == "__main__":
    print("🎯 HIERARCHICAL SWARM STREAMING DEMO")
    print("="*50)

    # Create agents and swarm
    agents = create_agents()
    swarm = HierarchicalSwarm(
        name="Research_and_Analysis_Swarm",
        description="A swarm that researches topics, analyzes information, and creates summaries",
        agents=agents,
        max_loops=1,
        verbose=True,
        director_model_name="gpt-4o-mini",
    )

    # Define task
    task = """
    Research the impact of artificial intelligence on the job market in 2024.
    Analyze how different industries are being affected and provide insights
    on future trends. Create a comprehensive summary of your findings.
    """

    print(f"Task: {task.strip()}")

    # Create streaming callback
    streaming_callback = create_streaming_callback()

    print("\n🎬 EXECUTING WITH STREAMING CALLBACKS...")
    print("Watch real-time agent outputs below:\n")

    # Execute with streaming
    result = swarm.run(
        task=task,
        streaming_callback=streaming_callback,
    )

    print("\n🎉 EXECUTION COMPLETED!")
    print("\n📊 FINAL RESULT:")
    print("-" * 50)

    # Display final result
    if isinstance(result, dict):
        for key, value in result.items():
            print(f"\n{key}:")
            print(f"{value}")
    else:
        print(result)
