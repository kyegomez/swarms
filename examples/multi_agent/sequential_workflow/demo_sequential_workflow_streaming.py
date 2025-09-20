from typing import Callable
from swarms.structs.agent import Agent
from swarms.structs.sequential_workflow import SequentialWorkflow

def create_streaming_callback() -> Callable[[str, str, bool], None]:
    """
    Create a streaming callback that prints each chunk as it arrives, with no duplication or timestamps.
    """
    def streaming_callback(agent_name: str, chunk: str, is_final: bool):
        if chunk:
            print(chunk, end="", flush=True)
        if is_final:
            print()
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
            print_on=False,
        ),
        Agent(
            agent_name="Analysis_Agent",
            agent_description="Expert at analyzing data and drawing insights",
            system_prompt="You are an analysis expert. Break down complex information and provide clear insights.",
            model_name="gpt-4o-mini",
            max_loops=1,
            streaming_on=True,
            print_on=False,
        ),
    ]

if __name__ == "__main__":
    print("🎯 SEQUENTIAL WORKFLOW STREAMING DEMO")
    print("=" * 50)

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

    task = "What are the latest advancements in AI?"

    print(f"Task: {task.strip()}")

    streaming_callback = create_streaming_callback()

    print("\n🎬 EXECUTING WITH STREAMING CALLBACKS...")
    print("Watch real-time agent outputs below:\n")

    result = workflow.run(
        task=task,
        streaming_callback=streaming_callback,
    )

    print("\n🎉 EXECUTION COMPLETED!")
    print("\n📊 FINAL RESULT:")
    print("-" * 50)
    print(result)