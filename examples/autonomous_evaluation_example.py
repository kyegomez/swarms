"""
Example demonstrating the autonomous evaluation feature for AutoSwarmBuilder.

This example shows how to use the enhanced AutoSwarmBuilder with autonomous evaluation
that iteratively improves agent performance through feedback loops.
"""

from swarms.structs.auto_swarm_builder import (
    AutoSwarmBuilder,
    IterativeImprovementConfig,
)
from dotenv import load_dotenv

load_dotenv()


def main():
    """Demonstrate autonomous evaluation in AutoSwarmBuilder"""
    
    # Configure the evaluation process
    eval_config = IterativeImprovementConfig(
        max_iterations=3,  # Maximum 3 improvement iterations
        improvement_threshold=0.1,  # Stop if improvement < 10%
        evaluation_dimensions=[
            "accuracy",
            "helpfulness", 
            "coherence",
            "instruction_adherence"
        ],
        use_judge_agent=True,
        store_all_iterations=True,
    )
    
    # Create AutoSwarmBuilder with autonomous evaluation enabled
    swarm = AutoSwarmBuilder(
        name="AutonomousResearchSwarm",
        description="A self-improving swarm for research tasks",
        verbose=True,
        max_loops=1,
        enable_evaluation=True,
        evaluation_config=eval_config,
    )
    
    # Define a research task
    task = """
    Research and analyze the current state of autonomous vehicle technology,
    including key players, recent breakthroughs, challenges, and future outlook.
    Provide a comprehensive report with actionable insights.
    """
    
    print("=" * 80)
    print("AUTONOMOUS EVALUATION DEMO")
    print("=" * 80)
    print(f"Task: {task}")
    print("\nStarting autonomous evaluation process...")
    print("The swarm will iteratively improve based on evaluation feedback.\n")
    
    # Run the swarm with autonomous evaluation
    try:
        result = swarm.run(task)
        
        print("\n" + "=" * 80)
        print("FINAL RESULT")
        print("=" * 80)
        print(result)
        
        # Display evaluation results
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        
        evaluation_results = swarm.get_evaluation_results()
        print(f"Total iterations completed: {len(evaluation_results)}")
        
        for i, eval_result in enumerate(evaluation_results):
            print(f"\n--- Iteration {i+1} ---")
            overall_score = sum(eval_result.evaluation_scores.values()) / len(eval_result.evaluation_scores)
            print(f"Overall Score: {overall_score:.3f}")
            
            print("Dimension Scores:")
            for dimension, score in eval_result.evaluation_scores.items():
                print(f"  {dimension}: {score:.3f}")
                
            print(f"Strengths: {len(eval_result.strengths)} identified")
            print(f"Weaknesses: {len(eval_result.weaknesses)} identified") 
            print(f"Suggestions: {len(eval_result.suggestions)} provided")
        
        # Show best iteration
        best_iteration = swarm.get_best_iteration()
        if best_iteration:
            best_score = sum(best_iteration.evaluation_scores.values()) / len(best_iteration.evaluation_scores)
            print(f"\nBest performing iteration: {best_iteration.iteration} (Score: {best_score:.3f})")
            
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        print("This might be due to missing API keys or network issues.")


def basic_example():
    """Show basic usage without evaluation for comparison"""
    print("\n" + "=" * 80)
    print("BASIC MODE (No Evaluation)")
    print("=" * 80)
    
    # Basic swarm without evaluation
    basic_swarm = AutoSwarmBuilder(
        name="BasicResearchSwarm",
        description="A basic swarm for research tasks",
        verbose=True,
        enable_evaluation=False,  # Evaluation disabled
    )
    
    task = "Write a brief summary of renewable energy trends."
    
    try:
        result = basic_swarm.run(task)
        print("Basic Result (no iterative improvement):")
        print(result)
        
    except Exception as e:
        print(f"Error during basic execution: {str(e)}")


if __name__ == "__main__":
    main()
    basic_example()