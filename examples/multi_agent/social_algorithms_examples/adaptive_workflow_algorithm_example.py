"""
Adaptive Workflow Social Algorithm Example

This example demonstrates an adaptive workflow algorithm where the process
dynamically changes based on intermediate results and feedback. The workflow
adapts its strategy, agent roles, and execution path based on what it learns.
"""

from swarms import Agent
from swarms.structs.social_algorithms import SocialAlgorithms


# Create versatile agents that can adapt their roles
adaptive_agent1 = Agent(
    agent_name="Adaptive_Agent_1",
    system_prompt="You are a versatile agent that can adapt your role based on the situation. You can be a researcher, analyst, creator, or reviewer as needed. You excel at switching contexts and approaches.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

adaptive_agent2 = Agent(
    agent_name="Adaptive_Agent_2",
    system_prompt="You are a versatile agent that can adapt your role based on the situation. You can be a researcher, analyst, creator, or reviewer as needed. You excel at switching contexts and approaches.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

adaptive_agent3 = Agent(
    agent_name="Adaptive_Agent_3",
    system_prompt="You are a versatile agent that can adapt your role based on the situation. You can be a researcher, analyst, creator, or reviewer as needed. You excel at switching contexts and approaches.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

workflow_controller = Agent(
    agent_name="Workflow_Controller",
    system_prompt="You are a workflow controller that monitors progress, analyzes results, and dynamically adjusts the workflow strategy. You make decisions about what to do next based on what you learn.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

quality_monitor = Agent(
    agent_name="Quality_Monitor",
    system_prompt="You are a quality monitor that continuously assesses the quality and progress of work. You provide feedback that helps the workflow controller make adaptive decisions.",
    model_name="gpt-4o-mini",
    max_loops=1,
)


def adaptive_workflow_algorithm(agents, task, **kwargs):
    """
    An adaptive workflow algorithm that dynamically changes its strategy
    based on intermediate results and feedback.
    """
    adaptive_agents = agents[:-2]  # First 3 agents are adaptive
    controller_agent = agents[-2]  # Second to last is controller
    monitor_agent = agents[-1]  # Last is quality monitor

    max_phases = kwargs.get("max_phases", 6)
    quality_threshold = kwargs.get("quality_threshold", 7.0)

    # Initialize workflow state
    workflow_history = []
    current_strategy = "exploration"  # Start with exploration
    phase_count = 0
    accumulated_knowledge = []
    quality_scores = []

    # Phase 1: Initial Assessment and Strategy Selection
    controller_agent.run(f"Starting adaptive workflow for: {task}")

    assessment_prompt = f"""
    Analyze this task and determine the best initial strategy:
    Task: {task}
    
    Available strategies:
    1. exploration - Gather information and explore possibilities
    2. analysis - Deep dive into specific aspects
    3. creation - Generate solutions and prototypes
    4. validation - Test and validate approaches
    5. synthesis - Combine and integrate findings
    
    Choose the most appropriate initial strategy and explain why.
    """

    initial_strategy = controller_agent.run(assessment_prompt)
    workflow_history.append(
        {
            "phase": 0,
            "action": "initial_assessment",
            "strategy": initial_strategy,
            "agent": controller_agent.agent_name,
        }
    )

    # Adaptive workflow phases
    while phase_count < max_phases:
        phase_count += 1
        controller_agent.run(f"Starting adaptive phase {phase_count}")

        # Quality Monitor assesses current state
        if accumulated_knowledge:
            quality_prompt = f"""
            Assess the current quality and progress:
            
            Task: {task}
            Current Strategy: {current_strategy}
            Accumulated Knowledge: {accumulated_knowledge[-3:] if len(accumulated_knowledge) >= 3 else accumulated_knowledge}
            
            Rate the quality (1-10) and provide specific feedback on:
            1. Completeness
            2. Accuracy
            3. Relevance
            4. Progress toward goal
            
            What should be the focus for the next phase?
            """

            quality_assessment = monitor_agent.run(quality_prompt)

            # Extract quality score (simplified)
            quality_score = 7.0  # Default
            if "rate the quality" in quality_assessment.lower():
                try:
                    import re

                    scores = re.findall(
                        r"(\d+(?:\.\d+)?)", quality_assessment
                    )
                    if scores:
                        quality_score = float(scores[0])
                except:
                    pass

            quality_scores.append(quality_score)
        else:
            quality_score = 5.0  # Default for first phase
            quality_scores.append(quality_score)
            quality_assessment = (
                "Initial phase - no previous work to assess"
            )

        # Workflow Controller decides next action based on quality and progress
        decision_prompt = f"""
        Based on the quality assessment, decide the next phase strategy:
        
        Current Phase: {phase_count}
        Current Strategy: {current_strategy}
        Quality Score: {quality_score}
        Quality Assessment: {quality_assessment}
        Previous Phases: {len(workflow_history)} phases completed
        
        Available strategies:
        1. exploration - Gather more information
        2. analysis - Deep dive into specific aspects
        3. creation - Generate solutions
        4. validation - Test approaches
        5. synthesis - Combine findings
        6. refinement - Improve existing work
        7. completion - Finalize and conclude
        
        Choose the next strategy and explain your reasoning.
        Consider the quality score and what would be most beneficial.
        """

        strategy_decision = controller_agent.run(decision_prompt)

        # Extract new strategy (simplified)
        new_strategy = current_strategy  # Default to current
        if "exploration" in strategy_decision.lower():
            new_strategy = "exploration"
        elif "analysis" in strategy_decision.lower():
            new_strategy = "analysis"
        elif "creation" in strategy_decision.lower():
            new_strategy = "creation"
        elif "validation" in strategy_decision.lower():
            new_strategy = "validation"
        elif "synthesis" in strategy_decision.lower():
            new_strategy = "synthesis"
        elif "refinement" in strategy_decision.lower():
            new_strategy = "refinement"
        elif "completion" in strategy_decision.lower():
            new_strategy = "completion"

        current_strategy = new_strategy

        # Execute the chosen strategy
        if current_strategy == "exploration":
            # Assign exploration roles to adaptive agents
            exploration_results = []
            for i, agent in enumerate(adaptive_agents):
                exploration_prompt = f"""
                As an adaptive agent, explore this aspect of the task: {task}
                
                Focus on:
                - New information and insights
                - Different perspectives and approaches
                - Potential opportunities and challenges
                - Relevant examples and case studies
                
                Provide your exploration findings.
                """

                result = agent.run(exploration_prompt)
                exploration_results.append(
                    {"agent": agent.agent_name, "exploration": result}
                )

            accumulated_knowledge.extend(exploration_results)
            workflow_history.append(
                {
                    "phase": phase_count,
                    "strategy": current_strategy,
                    "results": exploration_results,
                    "quality_score": quality_score,
                }
            )

        elif current_strategy == "analysis":
            # Assign analysis roles
            analysis_results = []
            for i, agent in enumerate(adaptive_agents):
                analysis_prompt = f"""
                As an adaptive agent, analyze the accumulated knowledge:
                
                Previous Knowledge: {accumulated_knowledge[-2:] if len(accumulated_knowledge) >= 2 else accumulated_knowledge}
                Task: {task}
                
                Focus on:
                - Identifying patterns and connections
                - Analyzing strengths and weaknesses
                - Evaluating different approaches
                - Drawing insights and conclusions
                
                Provide your analysis.
                """

                result = agent.run(analysis_prompt)
                analysis_results.append(
                    {"agent": agent.agent_name, "analysis": result}
                )

            accumulated_knowledge.extend(analysis_results)
            workflow_history.append(
                {
                    "phase": phase_count,
                    "strategy": current_strategy,
                    "results": analysis_results,
                    "quality_score": quality_score,
                }
            )

        elif current_strategy == "creation":
            # Assign creation roles
            creation_results = []
            for i, agent in enumerate(adaptive_agents):
                creation_prompt = f"""
                As an adaptive agent, create solutions based on the accumulated knowledge:
                
                Knowledge Base: {accumulated_knowledge[-3:] if len(accumulated_knowledge) >= 3 else accumulated_knowledge}
                Task: {task}
                
                Focus on:
                - Generating concrete solutions
                - Creating prototypes or detailed plans
                - Developing implementation strategies
                - Proposing specific actions
                
                Provide your creative solutions.
                """

                result = agent.run(creation_prompt)
                creation_results.append(
                    {"agent": agent.agent_name, "creation": result}
                )

            accumulated_knowledge.extend(creation_results)
            workflow_history.append(
                {
                    "phase": phase_count,
                    "strategy": current_strategy,
                    "results": creation_results,
                    "quality_score": quality_score,
                }
            )

        elif current_strategy == "validation":
            # Assign validation roles
            validation_results = []
            for i, agent in enumerate(adaptive_agents):
                validation_prompt = f"""
                As an adaptive agent, validate the proposed solutions:
                
                Proposed Solutions: {accumulated_knowledge[-2:] if len(accumulated_knowledge) >= 2 else accumulated_knowledge}
                Task: {task}
                
                Focus on:
                - Testing feasibility and viability
                - Identifying potential issues and risks
                - Evaluating effectiveness and impact
                - Suggesting improvements and modifications
                
                Provide your validation results.
                """

                result = agent.run(validation_prompt)
                validation_results.append(
                    {"agent": agent.agent_name, "validation": result}
                )

            accumulated_knowledge.extend(validation_results)
            workflow_history.append(
                {
                    "phase": phase_count,
                    "strategy": current_strategy,
                    "results": validation_results,
                    "quality_score": quality_score,
                }
            )

        elif current_strategy == "synthesis":
            # Assign synthesis roles
            synthesis_results = []
            for i, agent in enumerate(adaptive_agents):
                synthesis_prompt = f"""
                As an adaptive agent, synthesize the accumulated knowledge:
                
                All Knowledge: {accumulated_knowledge[-4:] if len(accumulated_knowledge) >= 4 else accumulated_knowledge}
                Task: {task}
                
                Focus on:
                - Combining and integrating findings
                - Creating coherent solutions
                - Resolving conflicts and contradictions
                - Building comprehensive approaches
                
                Provide your synthesis.
                """

                result = agent.run(synthesis_prompt)
                synthesis_results.append(
                    {"agent": agent.agent_name, "synthesis": result}
                )

            accumulated_knowledge.extend(synthesis_results)
            workflow_history.append(
                {
                    "phase": phase_count,
                    "strategy": current_strategy,
                    "results": synthesis_results,
                    "quality_score": quality_score,
                }
            )

        elif current_strategy == "refinement":
            # Assign refinement roles
            refinement_results = []
            for i, agent in enumerate(adaptive_agents):
                refinement_prompt = f"""
                As an adaptive agent, refine the existing work:
                
                Current Work: {accumulated_knowledge[-2:] if len(accumulated_knowledge) >= 2 else accumulated_knowledge}
                Task: {task}
                Quality Issues: {quality_assessment}
                
                Focus on:
                - Improving quality and completeness
                - Addressing identified issues
                - Enhancing clarity and precision
                - Strengthening weak areas
                
                Provide your refinements.
                """

                result = agent.run(refinement_prompt)
                refinement_results.append(
                    {"agent": agent.agent_name, "refinement": result}
                )

            accumulated_knowledge.extend(refinement_results)
            workflow_history.append(
                {
                    "phase": phase_count,
                    "strategy": current_strategy,
                    "results": refinement_results,
                    "quality_score": quality_score,
                }
            )

        elif current_strategy == "completion":
            # Final completion phase
            completion_prompt = f"""
            Create the final deliverable based on all accumulated work:
            
            Task: {task}
            Complete Knowledge Base: {accumulated_knowledge}
            Workflow History: {len(workflow_history)} phases
            
            Create a comprehensive final solution that incorporates all the work done.
            """

            final_deliverable = controller_agent.run(
                completion_prompt
            )
            workflow_history.append(
                {
                    "phase": phase_count,
                    "strategy": current_strategy,
                    "results": [
                        {
                            "agent": controller_agent.agent_name,
                            "final_deliverable": final_deliverable,
                        }
                    ],
                    "quality_score": quality_score,
                }
            )
            break

        # Check if we should continue or conclude
        if quality_score >= quality_threshold and phase_count >= 3:
            controller_agent.run(
                f"Quality threshold reached ({quality_score} >= {quality_threshold}). Considering completion."
            )
            if current_strategy != "completion":
                current_strategy = "completion"
                continue

    # Final workflow summary
    summary_prompt = f"""
    Provide a summary of the adaptive workflow process:
    
    Task: {task}
    Total Phases: {phase_count}
    Final Strategy: {current_strategy}
    Quality Scores: {quality_scores}
    Workflow History: {len(workflow_history)} phases
    
    Summarize the adaptive process and its effectiveness.
    """

    workflow_summary = controller_agent.run(summary_prompt)

    return {
        "task": task,
        "workflow_history": workflow_history,
        "accumulated_knowledge": accumulated_knowledge,
        "quality_scores": quality_scores,
        "final_strategy": current_strategy,
        "total_phases": phase_count,
        "workflow_summary": workflow_summary,
        "algorithm_type": "adaptive_workflow",
    }


# Adaptive Workflow Algorithm
social_alg = SocialAlgorithms(
    name="Adaptive-Workflow-Algorithm",
    description="Adaptive workflow algorithm that dynamically changes strategy based on results",
    agents=[
        adaptive_agent1,
        adaptive_agent2,
        adaptive_agent3,
        workflow_controller,
        quality_monitor,
    ],
    social_algorithm=adaptive_workflow_algorithm,
    verbose=True,
    max_execution_time=1200,  # 20 minutes for adaptive workflow
)

if __name__ == "__main__":
    result = social_alg.run(
        "Develop a comprehensive digital transformation strategy for a mid-size company",
        algorithm_args={"max_phases": 6, "quality_threshold": 7.0},
    )

    print("=== ADAPTIVE WORKFLOW ALGORITHM RESULTS ===")
    print(f"Task: {result.final_outputs['task']}")
    print(f"Total Phases: {result.final_outputs['total_phases']}")
    print(f"Final Strategy: {result.final_outputs['final_strategy']}")
    print(
        f"Average Quality Score: {sum(result.final_outputs['quality_scores'])/len(result.final_outputs['quality_scores']):.2f}"
    )
    print(f"Execution Time: {result.execution_time:.2f} seconds")
    print(f"Total Communication Steps: {result.total_steps}")

    print("\n=== WORKFLOW PHASES ===")
    for phase in result.final_outputs["workflow_history"]:
        print(
            f"Phase {phase['phase']}: {phase['strategy']} (Quality: {phase.get('quality_score', 'N/A')})"
        )
        if "results" in phase:
            print(f"  Results: {len(phase['results'])} outputs")
        print()

    print("\n=== QUALITY PROGRESSION ===")
    for i, score in enumerate(result.final_outputs["quality_scores"]):
        print(f"Phase {i+1}: {score}")

    print("\n=== WORKFLOW SUMMARY ===")
    print(result.final_outputs["workflow_summary"][:500] + "...")
