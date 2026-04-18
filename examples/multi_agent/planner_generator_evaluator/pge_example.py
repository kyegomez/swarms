"""
Planner-Generator-Evaluator Harness — Basic Example

Demonstrates the PGE harness with default agents. The Planner expands
the prompt into a plan, the Generator executes each step, and the
Evaluator scores output against criteria in an iterative feedback loop.

To run:
    python examples/multi_agent/planner_generator_evaluator/pge_example.py
"""

from swarms import PlannerGeneratorEvaluator

if __name__ == "__main__":
    harness = PlannerGeneratorEvaluator(
        model_name="gpt-4.1",
        max_steps=3,
        max_retries_per_step=2,
        output_type="final",
        verbose=True,
    )

    result = harness.run(
        "Write a comprehensive guide on the benefits and risks of intermittent fasting"
    )

    print(result)

    print(
        f"\nSteps completed: {harness.last_result.total_steps_completed}"
    )
    print(f"Total retries: {harness.last_result.total_retries}")
    print(f"Duration: {harness.last_result.total_duration:.1f}s")
    print(f"Shared state file: {harness.last_result.output_path}")
