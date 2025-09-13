from swarms.structs.agent import Agent
from swarms.structs.sequential_workflow import SequentialWorkflow

def run_workflow_with_streaming_callback(task, streaming_callback):
    """
    Run a sequential workflow with two agents and a streaming callback.

    Args:
        task (str): The task to process through the workflow.
        streaming_callback (callable): Function to handle streaming output.

    Returns:
        The final result from the workflow.
    """

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

# ## REGULAR STREAMING CALLBACK
#     def streaming_callback(token):
#         print(token, end="", flush=True)

#     run_workflow_with_streaming_callback(
#         task="What are the latest advancements in AI?",
#         streaming_callback=streaming_callback,
#     )


# ## CUSTOM BUFFERING STREAMING_CALLBACK BASED ON DEV PREFERED 
#     buffer = []
#     def streaming_callback(token):
#         buffer.append(token)
#         # Print in bigger chunks (e.g., every 20 tokens or on final flush)
#         if len(buffer) >= 20 or token.endswith("\n"):
#             print("".join(buffer), end="", flush=True)
#             buffer.clear()
#         # Optionally, you could add a flush at the end of the run if needed

#     run_workflow_with_streaming_callback(
#         task="What are the latest advancements in AI?",
#         streaming_callback=streaming_callback,
#     )


## NO ADDED STREAMING_CALLBACK
    run_workflow_with_streaming_callback(
        task="What are the latest advancements in AI?",
        streaming_callback=None,
    )