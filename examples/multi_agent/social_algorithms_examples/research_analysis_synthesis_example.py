from typing import List, Dict, Any
from swarms import Agent
from swarms.structs.social_algorithms import SocialAlgorithms


def research_analysis_synthesis_algorithm(
    agents: List[Agent], task: str, **kwargs
) -> Dict[str, Any]:
    """
    A sequential social algorithm where agents work in a research -> analysis -> synthesis pattern.

    Args:
        agents: List of agents participating in the algorithm
        task: The task to be processed
        **kwargs: Additional keyword arguments

    Returns:
        Dict containing the results from each agent
    """
    if len(agents) < 3:
        raise ValueError("This algorithm requires at least 3 agents")

    # Agent 1: Research
    research_agent = agents[0]
    research_prompt = (
        f"Research and gather comprehensive information about: {task}"
    )
    research_result = research_agent.run(research_prompt)

    # Agent 2: Analysis
    analysis_agent = agents[1]
    analysis_prompt = f"Analyze the following research findings and identify key insights:\n\n{research_result}"
    analysis_result = analysis_agent.run(analysis_prompt)

    # Agent 3: Synthesis
    synthesis_agent = agents[2]
    synthesis_prompt = f"Based on the research and analysis, create a comprehensive synthesis:\n\nResearch: {research_result}\n\nAnalysis: {analysis_result}"
    synthesis_result = synthesis_agent.run(synthesis_prompt)

    return {
        "research": research_result,
        "analysis": analysis_result,
        "synthesis": synthesis_result,
        "task": task,
    }


# Create agents
researcher = Agent(
    agent_name="Researcher",
    system_prompt="You are a research specialist focused on gathering comprehensive information.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

analyst = Agent(
    agent_name="Analyst",
    system_prompt="You are an analytical specialist focused on interpreting and analyzing data.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

synthesizer = Agent(
    agent_name="Synthesizer",
    system_prompt="You are a synthesis specialist focused on combining information into coherent outputs.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Create and run the social algorithm
social_alg = SocialAlgorithms(
    name="Research-Analysis-Synthesis",
    description="Sequential research, analysis, and synthesis workflow",
    agents=[researcher, analyst, synthesizer],
    social_algorithm=research_analysis_synthesis_algorithm,
    verbose=True,
)

result = social_alg.run(
    "The future of artificial intelligence in healthcare"
)
