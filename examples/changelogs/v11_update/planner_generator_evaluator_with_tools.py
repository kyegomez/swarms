from swarms import PlannerGeneratorEvaluator


def run_python(code: str) -> str:
    """Execute Python code and return stdout."""
    ...


harness = PlannerGeneratorEvaluator(
    generator_tools=[run_python],
    score_threshold=0.75,
)
result = harness.run("Implement and test a red-black tree in Python")
print(result)
