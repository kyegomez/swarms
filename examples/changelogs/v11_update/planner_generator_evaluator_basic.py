from swarms import PlannerGeneratorEvaluator

harness = PlannerGeneratorEvaluator(
    planner_model_name="gpt-5.4",
    generator_model_name="claude-sonnet-4-6",
    evaluator_model_name="claude-sonnet-4-6",
    score_threshold=0.8,  # minimum score to proceed
    max_retries=3,  # max Generator retries per step
    shared_file_path="./workspace/pge_shared.md",
)

result = harness.run(
    "Build a Python CLI tool that converts CSV to Parquet"
)
print(result)
