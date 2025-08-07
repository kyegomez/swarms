"""
Custom example demonstrating CouncilAsAJudge with specific configurations.

This example shows how to use the CouncilAsAJudge with different output types,
custom worker configurations, and focused evaluation scenarios.
"""

from swarms.structs.council_judge import CouncilAsAJudge


def evaluate_with_final_output():
    """
    Evaluate a response and return only the final aggregated result.
    """
    council = CouncilAsAJudge(
        name="Final Output Council",
        model_name="gpt-4o-mini",
        output_type="final",
        max_workers=2,
    )

    task = """
    Task: Write a brief explanation of climate change for middle school students.
    
    Response: Climate change is when the Earth's temperature gets warmer over time. This happens because of gases like carbon dioxide that trap heat in our atmosphere, kind of like a blanket around the Earth. Human activities like burning fossil fuels (gas, oil, coal) and cutting down trees are making this problem worse. The effects include melting ice caps, rising sea levels, more extreme weather like hurricanes and droughts, and changes in animal habitats. We can help by using renewable energy like solar and wind power, driving less, and planting trees. It's important for everyone to work together to reduce our impact on the environment.
    """

    return council.run(task=task)


def evaluate_with_conversation_output():
    """
    Evaluate a response and return the full conversation history.
    """
    council = CouncilAsAJudge(
        name="Conversation Council",
        model_name="gpt-4o-mini",
        output_type="conversation",
        max_workers=3,
    )

    task = """
    Task: Provide advice on how to start a small business.
    
    Response: Starting a small business requires careful planning and preparation. First, identify a market need and develop a unique value proposition. Conduct thorough market research to understand your competition and target audience. Create a detailed business plan that includes financial projections, marketing strategies, and operational procedures. Secure funding through savings, loans, or investors. Choose the right legal structure (sole proprietorship, LLC, corporation) and register your business with the appropriate authorities. Set up essential systems like accounting, inventory management, and customer relationship management. Build a strong online presence through a website and social media. Network with other entrepreneurs and join local business groups. Start small and scale gradually based on customer feedback and market demand. Remember that success takes time, persistence, and the ability to adapt to changing circumstances.
    """

    return council.run(task=task)


def evaluate_with_minimal_workers():
    """
    Evaluate a response using minimal worker threads for resource-constrained environments.
    """
    council = CouncilAsAJudge(
        name="Minimal Workers Council",
        model_name="gpt-4o-mini",
        output_type="all",
        max_workers=1,
        random_model_name=False,
    )

    task = """
    Task: Explain the benefits of regular exercise.
    
    Response: Regular exercise offers numerous physical and mental health benefits. Physically, it strengthens muscles and bones, improves cardiovascular health, and helps maintain a healthy weight. Exercise boosts energy levels and improves sleep quality. It also enhances immune function, reducing the risk of chronic diseases like heart disease, diabetes, and certain cancers. Mentally, exercise releases endorphins that reduce stress and anxiety while improving mood and cognitive function. It can help with depression and boost self-confidence. Regular physical activity also promotes better posture, flexibility, and balance, reducing the risk of falls and injuries. Additionally, exercise provides social benefits when done with others, fostering connections and accountability. Even moderate activities like walking, swimming, or cycling for 30 minutes most days can provide significant health improvements.
    """

    return council.run(task=task)


def main():
    """
    Main function demonstrating different CouncilAsAJudge configurations.
    """
    configurations = [
        ("Final Output Only", evaluate_with_final_output),
        ("Full Conversation", evaluate_with_conversation_output),
        ("Minimal Workers", evaluate_with_minimal_workers),
    ]

    results = {}

    for config_name, evaluation_func in configurations:
        print(f"\n{'='*60}")
        print(f"Configuration: {config_name}")
        print(f"{'='*60}")

        try:
            result = evaluation_func()
            results[config_name] = result
            print(f"✅ {config_name} evaluation completed!")

            # Show a preview of the result
            if isinstance(result, str):
                preview = (
                    result[:200] + "..."
                    if len(result) > 200
                    else result
                )
                print(f"Preview: {preview}")
            else:
                print(f"Result type: {type(result)}")

        except Exception as e:
            print(f"❌ {config_name} evaluation failed: {str(e)}")
            results[config_name] = None

    return results


if __name__ == "__main__":
    # Run all configuration examples
    all_results = main()

    # Display final summary
    print(f"\n{'='*60}")
    print("CONFIGURATION SUMMARY")
    print(f"{'='*60}")

    successful_configs = sum(
        1 for result in all_results.values() if result is not None
    )
    total_configs = len(all_results)

    print(
        f"Successful evaluations: {successful_configs}/{total_configs}"
    )

    for config_name, result in all_results.items():
        status = "✅ Success" if result else "❌ Failed"
        print(f"{config_name}: {status}")
