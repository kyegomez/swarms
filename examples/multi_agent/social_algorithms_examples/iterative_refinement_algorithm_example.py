from swarms import Agent
from swarms.structs.social_algorithms import SocialAlgorithms


# Create agents with different expertise areas
content_creator = Agent(
    agent_name="Content_Creator",
    system_prompt="You are a content creation specialist focused on generating high-quality, engaging content. You excel at creating initial drafts and creative content.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

technical_reviewer = Agent(
    agent_name="Technical_Reviewer",
    system_prompt="You are a technical reviewer focused on accuracy, technical correctness, and implementation feasibility. You identify technical issues and suggest improvements.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

style_editor = Agent(
    agent_name="Style_Editor",
    system_prompt="You are a style and language editor focused on clarity, readability, and professional presentation. You improve writing style and structure.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

quality_assessor = Agent(
    agent_name="Quality_Assessor",
    system_prompt="You are a quality assessor focused on overall quality, completeness, and meeting requirements. You provide final quality checks and approval.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

coordinator = Agent(
    agent_name="Coordinator",
    system_prompt="You are a project coordinator focused on managing the refinement process, tracking progress, and ensuring all improvements are integrated effectively.",
    model_name="gpt-4o-mini",
    max_loops=1,
)


def iterative_refinement_algorithm(agents, task, **kwargs):
    """
    An iterative refinement algorithm where agents work in multiple rounds,
    each building upon and improving the previous round's output.
    """
    content_agent = agents[0]
    technical_agent = agents[1]
    style_agent = agents[2]
    quality_agent = agents[3]
    coordinator_agent = agents[4]

    max_loops = kwargs.get("max_loops", 4)
    quality_threshold = kwargs.get("quality_threshold", 8.0)

    # Initialize tracking variables
    current_version = None
    iteration_history = []
    quality_scores = []

    # Phase 1: Initial Creation
    coordinator_agent.run(
        f"Starting iterative refinement process for: {task}"
    )

    initial_prompt = f"""
    Create an initial version of: {task}
    
    Focus on:
    1. Core content and ideas
    2. Basic structure and organization
    3. Key points and main arguments
    4. Initial implementation approach
    
    This is version 1.0 - we will refine it iteratively.
    """

    current_version = content_agent.run(initial_prompt)
    iteration_history.append(
        {
            "iteration": 1,
            "version": "1.0",
            "agent": content_agent.agent_name,
            "output": current_version,
            "type": "initial_creation",
        }
    )

    # Phase 2: Iterative Refinement Rounds
    for iteration in range(2, max_loops + 1):
        coordinator_agent.run(
            f"Starting iteration {iteration} of refinement process"
        )

        # Technical Review and Improvement
        technical_prompt = f"""
        Review and improve the technical aspects of this work (Version {iteration-1}.0):
        
        {current_version}
        
        Focus on:
        1. Technical accuracy and correctness
        2. Implementation feasibility
        3. Technical depth and detail
        4. Technical best practices
        
        Provide an improved version with technical enhancements.
        """

        technical_improved = technical_agent.run(technical_prompt)
        iteration_history.append(
            {
                "iteration": iteration,
                "version": f"{iteration-1}.1",
                "agent": technical_agent.agent_name,
                "output": technical_improved,
                "type": "technical_review",
            }
        )

        # Style and Language Improvement
        style_prompt = f"""
        Review and improve the style and presentation of this work (Version {iteration-1}.1):
        
        {technical_improved}
        
        Focus on:
        1. Clarity and readability
        2. Professional presentation
        3. Structure and organization
        4. Language and tone
        5. Flow and coherence
        
        Provide an improved version with style enhancements.
        """

        style_improved = style_agent.run(style_prompt)
        iteration_history.append(
            {
                "iteration": iteration,
                "version": f"{iteration-1}.2",
                "agent": style_agent.agent_name,
                "output": style_improved,
                "type": "style_improvement",
            }
        )

        # Quality Assessment
        quality_prompt = f"""
        Assess the quality of this work (Version {iteration-1}.2) on a scale of 1-10:
        
        {style_improved}
        
        Evaluate:
        1. Completeness (1-10)
        2. Accuracy (1-10)
        3. Clarity (1-10)
        4. Professionalism (1-10)
        5. Overall quality (1-10)
        
        Provide scores and specific feedback for improvement.
        If quality is below {quality_threshold}, suggest specific improvements.
        """

        quality_assessment = quality_agent.run(quality_prompt)

        # Extract quality score (simple heuristic)
        quality_score = 7.0  # Default
        if "overall quality" in quality_assessment.lower():
            try:
                # Look for numeric score in the assessment
                import re

                scores = re.findall(
                    r"(\d+(?:\.\d+)?)", quality_assessment
                )
                if scores:
                    quality_score = float(
                        scores[-1]
                    )  # Take the last number found
            except:
                pass

        quality_scores.append(quality_score)

        iteration_history.append(
            {
                "iteration": iteration,
                "version": f"{iteration-1}.2",
                "agent": quality_agent.agent_name,
                "output": quality_assessment,
                "type": "quality_assessment",
                "quality_score": quality_score,
            }
        )

        # Coordinator decides whether to continue or finalize
        if quality_score >= quality_threshold:
            coordinator_agent.run(
                f"Quality threshold reached ({quality_score} >= {quality_threshold}). Preparing for finalization."
            )
            break

        # Update current version for next iteration
        current_version = style_improved

        # Coordinator provides integration guidance
        integration_prompt = f"""
        As coordinator, provide guidance for the next iteration based on:
        
        Current Version: {current_version}
        Quality Assessment: {quality_assessment}
        Quality Score: {quality_score}
        Target Threshold: {quality_threshold}
        
        What should be the focus for the next iteration?
        """

        integration_guidance = coordinator_agent.run(
            integration_prompt
        )
        iteration_history.append(
            {
                "iteration": iteration,
                "version": f"{iteration-1}.3",
                "agent": coordinator_agent.agent_name,
                "output": integration_guidance,
                "type": "integration_guidance",
            }
        )

    # Phase 3: Final Integration and Approval
    final_prompt = f"""
    Create the final integrated version based on all iterations:
    
    Latest Version: {current_version}
    Quality Assessment: {quality_assessment}
    Iteration History: {len(iteration_history)} iterations completed
    
    Create a polished, final version that incorporates all improvements.
    """

    final_version = coordinator_agent.run(final_prompt)

    # Final quality check
    final_quality_prompt = f"""
    Perform final quality check on the completed work:
    
    {final_version}
    
    Provide final approval and summary of the refinement process.
    """

    final_approval = quality_agent.run(final_quality_prompt)

    return {
        "task": task,
        "final_version": final_version,
        "iteration_history": iteration_history,
        "quality_scores": quality_scores,
        "final_quality_score": (
            quality_scores[-1] if quality_scores else 0
        ),
        "total_iterations": len(iteration_history),
        "final_approval": final_approval,
        "algorithm_type": "iterative_refinement",
    }


# Iterative Refinement Algorithm
social_alg = SocialAlgorithms(
    name="Iterative-Refinement-Algorithm",
    description="Iterative refinement algorithm with multiple rounds of improvement",
    agents=[
        content_creator,
        technical_reviewer,
        style_editor,
        quality_assessor,
        coordinator,
    ],
    social_algorithm=iterative_refinement_algorithm,
    verbose=True,
    max_execution_time=900,  # 15 minutes for multiple iterations
)

if __name__ == "__main__":
    result = social_alg.run(
        "Create a comprehensive technical documentation for a new API",
        algorithm_args={
            "max_loops": 4,
            "quality_threshold": 8.0,
        },
    )

    print("=== ITERATIVE REFINEMENT ALGORITHM RESULTS ===")
    print(f"Task: {result.final_outputs['task']}")
    print(
        f"Total Iterations: {result.final_outputs['total_iterations']}"
    )
    print(
        f"Final Quality Score: {result.final_outputs['final_quality_score']}"
    )
    print(f"Execution Time: {result.execution_time:.2f} seconds")
    print(f"Total Communication Steps: {result.total_steps}")

    print("\n=== ITERATION HISTORY ===")
    for i, iteration in enumerate(
        result.final_outputs["iteration_history"]
    ):
        print(
            f"Iteration {iteration['iteration']} - {iteration['agent']} ({iteration['type']})"
        )
        if "quality_score" in iteration:
            print(f"  Quality Score: {iteration['quality_score']}")
        print(f"  Output: {iteration['output'][:100]}...")
        print()

    print("\n=== FINAL VERSION ===")
    print(result.final_outputs["final_version"][:500] + "...")
