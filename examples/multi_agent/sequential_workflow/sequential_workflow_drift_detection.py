"""
Example: SequentialWorkflow with drift detection enabled.

Set drift_detection=True to have a judge agent score the final output's
semantic alignment with the original task after the pipeline completes.
A warning is logged when the score falls below drift_threshold (default 0.75).
"""

from swarms import Agent, SequentialWorkflow

researcher = Agent(
    agent_name="Researcher",
    system_prompt="""You are a research specialist. Given a topic, produce a
    concise factual summary covering the key points, major actors, and
    recent developments.""",
    model_name="claude-sonnet-4-5",
    max_loops=1,
    temperature=1,
    thinking_tokens=1024,
)

analyst = Agent(
    agent_name="Analyst",
    system_prompt="""You are an analytical expert. Given research notes,
    identify the most significant implications and surface three clear
    takeaways that directly address the original question.""",
    model_name="claude-sonnet-4-5",
    max_loops=1,
    temperature=1,
    thinking_tokens=1024,
)

writer = Agent(
    agent_name="Writer",
    system_prompt="""You are a professional writer. Given analytical takeaways,
    produce a polished, reader-friendly summary in 2-3 paragraphs that
    directly answers the original question.""",
    model_name="claude-sonnet-4-5",
    max_loops=1,
    temperature=1,
    thinking_tokens=1024,
)

wf = SequentialWorkflow(
    name="geopolitics-pipeline",
    agents=[researcher, analyst, writer],
    max_loops=1,
    drift_detection=True,
    drift_threshold=0.75,
    drift_model="claude-sonnet-4-5",
)

result = wf.run(
    "Summarize the geopolitical impact of rare earth mining in the Congo"
)
print(result)
