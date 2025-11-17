"""
Simple example demonstrating CouncilAsAJudge usage.

This example shows how to use the CouncilAsAJudge to evaluate a task response
across multiple dimensions including accuracy, helpfulness, harmlessness,
coherence, conciseness, and instruction adherence.
"""

from swarms.structs.council_as_judge import CouncilAsAJudge


def main():
    """
    Main function demonstrating CouncilAsAJudge usage.
    """
    # Initialize the council judge
    council = CouncilAsAJudge(
        name="Quality Evaluation Council",
        description="Evaluates response quality across multiple dimensions",
        model_name="gpt-4o-mini",
        max_workers=4,
    )

    # Example task with a response to evaluate
    task_with_response = """
    Task: Explain the concept of machine learning to a beginner.
    
    Response: Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It works by analyzing large amounts of data to identify patterns and make predictions or decisions. There are three main types: supervised learning (using labeled data), unsupervised learning (finding hidden patterns), and reinforcement learning (learning through trial and error). Machine learning is used in various applications like recommendation systems, image recognition, and natural language processing.
    """

    # Run the evaluation
    result = council.run(task=task_with_response)

    return result


if __name__ == "__main__":
    # Run the example
    evaluation_result = main()

    # Display the result
    print("Council Evaluation Complete!")
    print("=" * 50)
    print(evaluation_result)
