from swarms.structs.agent import Agent
from swarms.structs.sequential_workflow import SequentialWorkflow

def streaming_callback(agent_name: str, chunk: str, is_final: bool):
    if chunk:
        print(chunk, end="", flush=True)
    if is_final:
        print()

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

    workflow.run(
        task=task,
        streaming_callback=streaming_callback,
    )
