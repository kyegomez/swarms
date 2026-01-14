import json
from swarms import Agent
from swarms.structs.agent_grpo import AgenticGRPO


# Create a math-specialized agent for complex problems
math_system_prompt = """
You are an expert mathematician specializing in advanced mathematics including calculus, linear algebra, differential equations, and complex analysis.

Your task is to solve complex mathematical problems with rigorous mathematical reasoning and step-by-step solutions.

Guidelines:
- Show all intermediate steps and reasoning
- Use proper mathematical notation and terminology
- Verify your solutions when possible
- For calculus problems, show derivatives and integrals clearly
- For algebraic problems, show all algebraic manipulations
- For differential equations, show the complete solution process
- Always provide the final answer in a clear, unambiguous format

When solving problems:
1. Carefully analyze the problem structure
2. Identify the mathematical domain (calculus, algebra, etc.)
3. Apply appropriate theorems, formulas, or methods
4. Show all computational steps
5. Verify the solution if possible
6. State the final answer clearly

Always end your response with the final numerical or symbolic answer in a clear format."""

math_agent = Agent(
    agent_name="Advanced Math Specialist",
    agent_description="An expert mathematician specialized in solving complex mathematical problems including calculus, differential equations, and advanced algebra",
    system_prompt=math_system_prompt,
    model_name="anthropic/claude-sonnet-4-20250514",
    max_loops=1,
    temperature=0.2,  # Lower temperature for more precise mathematical reasoning
    verbose=False,
    top_p=None,
)

# Define a complex math problem
task = """
Solve the following differential equation:
dy/dx + 2y = 4e^(-2x)

with initial condition y(0) = 3.

Find the particular solution and evaluate y(1).
"""
correct_answers = [
    "0.406",
    "3e^(-2)",
    "3e^{-2}",
    "0.4056",
]  # Multiple acceptable answer formats

# Create AgenticGRPO instance
grpo = AgenticGRPO(
    name="Advanced Math Problem GRPO",
    description="GRPO for solving complex differential equations",
    agent=math_agent,
    n=5,
    correct_answers=correct_answers,
)

# Run the GRPO process
correct_responses = grpo.run(task=task)
print(f"Correct responses: {json.dumps(correct_responses, indent=4)}")

# Get all responses
all_responses = grpo.get_all()
print(f"All responses: {json.dumps(all_responses, indent=4)}")

# Results are now in:
# - correct_responses: List of responses that got the correct answer (rating == 1)
# - all_responses: List of all responses with uuid, task, answer, and rating
