"""
BatchedGridWorkflow Examples

This module demonstrates various usage patterns of the BatchedGridWorkflow
for parallel task execution across multiple agents.
"""

from swarms import Agent
from swarms.structs.batched_grid_workflow import BatchedGridWorkflow


def basic_batched_workflow():
    """Basic example of BatchedGridWorkflow with two agents."""
    # Create specialized agents
    writer_agent = Agent(
        model="gpt-4",
        system_prompt="You are a creative writer who specializes in storytelling and creative content.",
    )

    analyst_agent = Agent(
        model="gpt-4",
        system_prompt="You are a data analyst who specializes in research and analysis.",
    )

    # Create the workflow
    workflow = BatchedGridWorkflow(
        name="Content Creation Workflow",
        description="Parallel content creation and analysis",
        agents=[writer_agent, analyst_agent],
        max_loops=1,
    )

    # Define tasks for each agent
    tasks = [
        "Write a short story about a robot learning to paint",
        "Analyze the impact of AI on creative industries",
    ]

    # Execute the workflow
    result = workflow.run(tasks)
    return result


def multi_agent_research_workflow():
    """Example with multiple agents for research tasks."""
    # Create research agents with different specializations
    tech_agent = Agent(
        model="gpt-4",
        system_prompt="You are a technology researcher focused on emerging tech trends.",
    )

    business_agent = Agent(
        model="gpt-4",
        system_prompt="You are a business analyst specializing in market research and strategy.",
    )

    science_agent = Agent(
        model="gpt-4",
        system_prompt="You are a scientific researcher focused on breakthrough discoveries.",
    )

    # Create workflow with conversation tracking
    workflow = BatchedGridWorkflow(
        name="Multi-Domain Research",
        description="Parallel research across technology, business, and science",
        agents=[tech_agent, business_agent, science_agent],
        conversation_args={"message_id_on": True},
        max_loops=2,
    )

    # Research tasks
    tasks = [
        "Research the latest developments in quantum computing",
        "Analyze market trends in renewable energy sector",
        "Investigate recent breakthroughs in gene therapy",
    ]

    # Execute research workflow
    result = workflow.run(tasks)
    return result


def content_creation_workflow():
    """Example for content creation with iterative refinement."""
    # Create content creation agents
    blog_writer = Agent(
        model="gpt-4",
        system_prompt="You are a professional blog writer who creates engaging, informative content.",
    )

    social_media_manager = Agent(
        model="gpt-4",
        system_prompt="You are a social media expert who creates viral content and captions.",
    )

    editor_agent = Agent(
        model="gpt-4",
        system_prompt="You are an editor who reviews and improves content for clarity and impact.",
    )

    # Create workflow with multiple loops for refinement
    workflow = BatchedGridWorkflow(
        name="Content Creation Pipeline",
        description="Multi-stage content creation with review and refinement",
        agents=[blog_writer, social_media_manager, editor_agent],
        max_loops=3,
    )

    # Content creation tasks
    tasks = [
        "Write a comprehensive blog post about sustainable living",
        "Create engaging social media posts for a tech startup",
        "Review and improve the following content for maximum impact",
    ]

    # Execute content creation workflow
    result = workflow.run(tasks)
    return result


def problem_solving_workflow():
    """Example for collaborative problem solving."""
    # Create problem-solving agents
    technical_expert = Agent(
        model="gpt-4",
        system_prompt="You are a technical expert who analyzes problems from an engineering perspective.",
    )

    user_experience_designer = Agent(
        model="gpt-4",
        system_prompt="You are a UX designer who focuses on user needs and experience.",
    )

    business_strategist = Agent(
        model="gpt-4",
        system_prompt="You are a business strategist who considers market viability and business impact.",
    )

    # Create collaborative workflow
    workflow = BatchedGridWorkflow(
        name="Collaborative Problem Solving",
        description="Multi-perspective problem analysis and solution development",
        agents=[
            technical_expert,
            user_experience_designer,
            business_strategist,
        ],
        max_loops=1,
    )

    # Problem-solving tasks
    tasks = [
        "Analyze the technical feasibility of implementing a new feature",
        "Evaluate the user experience impact of the proposed changes",
        "Assess the business value and market potential of the solution",
    ]

    # Execute problem-solving workflow
    result = workflow.run(tasks)
    return result


def educational_workflow():
    """Example for educational content creation."""
    # Create educational agents
    math_tutor = Agent(
        model="gpt-4",
        system_prompt="You are a mathematics tutor who explains concepts clearly and provides examples.",
    )

    science_teacher = Agent(
        model="gpt-4",
        system_prompt="You are a science teacher who makes complex topics accessible and engaging.",
    )

    language_instructor = Agent(
        model="gpt-4",
        system_prompt="You are a language instructor who helps students improve their communication skills.",
    )

    # Create educational workflow
    workflow = BatchedGridWorkflow(
        name="Educational Content Creation",
        description="Parallel creation of educational materials across subjects",
        agents=[math_tutor, science_teacher, language_instructor],
        max_loops=1,
    )

    # Educational tasks
    tasks = [
        "Create a lesson plan for teaching calculus to high school students",
        "Develop an experiment to demonstrate photosynthesis",
        "Design a writing exercise to improve essay composition skills",
    ]

    # Execute educational workflow
    result = workflow.run(tasks)
    return result


# Example usage patterns
if __name__ == "__main__":
    # Run basic example
    basic_result = basic_batched_workflow()

    # Run research example
    research_result = multi_agent_research_workflow()

    # Run content creation example
    content_result = content_creation_workflow()

    # Run problem solving example
    problem_result = problem_solving_workflow()

    # Run educational example
    education_result = educational_workflow()
