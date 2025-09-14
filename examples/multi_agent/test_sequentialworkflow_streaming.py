from swarms.structs.agent import Agent
from swarms.structs.sequential_workflow import SequentialWorkflow

def streaming_callback(token):
    buffer.append(token)
    if len(buffer) >= 20 or token.endswith("\n"):
        print("".join(buffer), end="", flush=True)
        buffer.clear()

def run_workflow_with_streaming_callback(task, streaming_callback):

    agent1 = Agent(
        name="Research Agent",
        description="A research agent that can answer questions",
        model_name="gpt-4o",
        system_prompt=(
            "You are a ResearchAgent. Your task is to research and gather "
            "information about the given topic. Provide comprehensive research "
            "findings and key insights."
        ),
        max_loops=1,
        interactive=True,
        verbose=True,
    )

    agent2 = Agent(
        name="Analysis Agent",
        description="An analysis agent that draws conclusions from research",
        model_name="gpt-4o-mini",
        system_prompt=(
            "You are an AnalysisAgent. Your task is to analyze the research "
            "provided by the previous agent and draw meaningful conclusions. "
            "Provide detailed analysis and actionable insights."
        ),
        max_loops=1,
        interactive=True,
        verbose=True,
    )

    workflow = SequentialWorkflow(
        id="research_analysis_workflow",
        name="Research Analysis Workflow",
        description="A sequential workflow that researches and analyzes topics",
        agents=[agent1, agent2],
        max_loops=1,
        output_type="str",
        streaming_callback=streaming_callback,
        multi_agent_collab_prompt=True,
    )
    return workflow.run(task)

if __name__ == "__main__":

    buffer = []

    run_workflow_with_streaming_callback(
        task="What are the latest advancements in AI?",
        streaming_callback=streaming_callback,
    )