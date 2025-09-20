import time
from typing import Callable
from swarms.structs.agent import Agent
from swarms.structs.sequential_workflow import SequentialWorkflow

def create_streaming_callback() -> Callable[[str, str, bool], None]:
    """
    Create a streaming callback that shows live paragraph formation.
    """
    agent_buffers = {}
    paragraph_count = {}

    def streaming_callback(agent_name: str, chunk: str, is_final: bool):
        timestamp = time.strftime("%H:%M:%S")

        # Initialize buffers for new agents
        if agent_name not in agent_buffers:
            agent_buffers[agent_name] = ""
            paragraph_count[agent_name] = 1
            print(f"\n🎬 [{timestamp}] {agent_name} starting...")
            print("=" * 60)

        if chunk.strip():
            # Split chunk into tokens (words/punctuation)
            tokens = chunk.replace("\n", " \n ").split()

            for token in tokens:
                # Handle paragraph breaks
                if token == "\n":
                    if agent_buffers[agent_name].strip():
                        print(
                            f"\n📄 [{timestamp}] {agent_name} - Paragraph {paragraph_count[agent_name]} Complete:"
                        )
                        print(f"{agent_buffers[agent_name].strip()}")
                        print("=" * 60)
                        paragraph_count[agent_name] += 1
                        agent_buffers[agent_name] = ""
                else:
                    # Add token to buffer and show live accumulation
                    agent_buffers[agent_name] += token + " "
                    print(
                        f"\r[{timestamp}] {agent_name} | {agent_buffers[agent_name].strip()}",
                        end="",
                        flush=True,
                    )

        if is_final:
            print()  # New line after live updates
            # Print any remaining content as final paragraph
            if agent_buffers[agent_name].strip():
                print(
                    f"\n✅ [{timestamp}] {agent_name} COMPLETED - Final Paragraph:"
                )
                print(f"{agent_buffers[agent_name].strip()}")
                print()
            print(f"🎯 [{timestamp}] {agent_name} finished processing")
            print(f"📊 Total paragraphs processed: {paragraph_count[agent_name] - 1}")
            print("=" * 60)

    return streaming_callback

def create_agents():
    """Create specialized agents for the workflow."""
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
    ]

if __name__ == "__main__":
    print("🎯 SEQUENTIAL WORKFLOW STREAMING DEMO")
    print("=" * 50)

    # Create agents and workflow
    agents = create_agents()
    workflow = SequentialWorkflow(
        id="research_analysis_workflow",
        name="Research Analysis Workflow",
        description="A sequential workflow that researches and analyzes topics",
        agents=agents,
        max_loops=1,
        output_type="str",
        multi_agent_collab_prompt=True,
    )

    # Define task
    task = "What are the latest advancements in AI?"

    print(f"Task: {task.strip()}")

    # Create streaming callback
    streaming_callback = create_streaming_callback()

    print("\n🎬 EXECUTING WITH STREAMING CALLBACKS...")
    print("Watch real-time agent outputs below:\n")

    # Execute with streaming
    result = workflow.run(
        task=task,
        streaming_callback=streaming_callback,
    )

    print("\n🎉 EXECUTION COMPLETED!")
    print("\n📊 FINAL RESULT:")
    print("-" * 50)
    print(result)