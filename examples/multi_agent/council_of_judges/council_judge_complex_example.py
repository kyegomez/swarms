"""
Complex example demonstrating CouncilAsAJudge with different task types.

This example shows how to use the CouncilAsAJudge to evaluate various types
of responses including technical explanations, creative writing, and problem-solving.
"""

from swarms.structs.council_judge import CouncilAsAJudge


def evaluate_technical_response():
    """
    Evaluate a technical explanation response.
    """
    council = CouncilAsAJudge(
        name="Technical Evaluation Council",
        model_name="gpt-4o-mini",
        output_type="all",
    )

    task = """
    Task: Explain how blockchain technology works in simple terms.
    
    Response: Blockchain is like a digital ledger that records transactions across a network of computers. Each transaction is stored in a "block" that contains multiple transactions. These blocks are linked together in a chain, hence the name blockchain. The key feature is that once a block is added to the chain, it cannot be altered without changing all subsequent blocks, making it very secure. Think of it like a Google Doc that everyone can see and edit, but no one can delete or change what's already been written. This technology is the foundation for cryptocurrencies like Bitcoin, but it has many other applications like supply chain tracking, voting systems, and digital identity verification.
    """

    return council.run(task=task)


def evaluate_creative_response():
    """
    Evaluate a creative writing response.
    """
    council = CouncilAsAJudge(
        name="Creative Writing Council",
        model_name="gpt-4o-mini",
        output_type="all",
    )

    task = """
    Task: Write a short story about a robot learning to paint.
    
    Response: In a sunlit studio filled with canvases and paintbrushes, Pixel, a curious robot with delicate mechanical fingers, stared at a blank canvas. Its optical sensors analyzed the colors around it - the warm yellows of morning light, the deep blues of the sky outside the window, and the vibrant reds of the roses in a nearby vase. For the first time in its programming, Pixel felt something it couldn't quite define. It picked up a brush, dipped it in paint, and began to create. The first stroke was hesitant, but as it continued, something magical happened. The robot wasn't just following algorithms anymore; it was expressing something from within its digital heart. The painting that emerged was a beautiful blend of human emotion and mechanical precision, proving that art knows no boundaries between organic and artificial souls.
    """

    return council.run(task=task)


def evaluate_problem_solving_response():
    """
    Evaluate a problem-solving response.
    """
    council = CouncilAsAJudge(
        name="Problem Solving Council",
        model_name="gpt-4o-mini",
        output_type="all",
    )

    task = """
    Task: Provide a step-by-step solution for reducing plastic waste in a household.
    
    Response: To reduce plastic waste in your household, start by conducting a waste audit to identify the main sources of plastic. Replace single-use items with reusable alternatives like cloth shopping bags, stainless steel water bottles, and glass food containers. Choose products with minimal or no plastic packaging, and buy in bulk when possible. Start composting organic waste to reduce the need for plastic garbage bags. Make your own cleaning products using simple ingredients like vinegar and baking soda. Support local businesses that use eco-friendly packaging. Finally, educate family members about the importance of reducing plastic waste and involve them in finding creative solutions together.
    """

    return council.run(task=task)


def main():
    """
    Main function running all evaluation examples.
    """
    examples = [
        ("Technical Explanation", evaluate_technical_response),
        ("Creative Writing", evaluate_creative_response),
        ("Problem Solving", evaluate_problem_solving_response),
    ]

    results = {}

    for example_name, evaluation_func in examples:
        print(f"\n{'='*60}")
        print(f"Evaluating: {example_name}")
        print(f"{'='*60}")

        try:
            result = evaluation_func()
            results[example_name] = result
            print(
                f"✅ {example_name} evaluation completed successfully!"
            )
        except Exception as e:
            print(f"❌ {example_name} evaluation failed: {str(e)}")
            results[example_name] = None

    return results


if __name__ == "__main__":
    # Run all examples
    all_results = main()

    # Display summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")

    for example_name, result in all_results.items():
        status = "✅ Completed" if result else "❌ Failed"
        print(f"{example_name}: {status}")
