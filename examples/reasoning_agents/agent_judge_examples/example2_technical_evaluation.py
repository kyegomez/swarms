from swarms import AgentJudge

# Initialize the agent judge with custom evaluation criteria
judge = AgentJudge(
    agent_name="technical-judge",
    model_name="gpt-4",
    max_loops=1,
    evaluation_criteria={
        "accuracy": 0.4,
        "completeness": 0.3,
        "clarity": 0.2,
        "logic": 0.1,
    },
)

# Example technical agent output to evaluate
technical_output = "To solve the quadratic equation x² + 5x + 6 = 0, we can use the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a. Here, a=1, b=5, c=6. Substituting: x = (-5 ± √(25 - 24)) / 2 = (-5 ± √1) / 2 = (-5 ± 1) / 2. So x = -2 or x = -3."

# Run evaluation with context building
evaluations = judge.run(task=technical_output)
print(evaluations)
