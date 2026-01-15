import json
from swarms import Agent
from swarms.structs.agent_grpo import AgenticGRPO


# Create a physics-specialized agent for complex problems
physics_system_prompt = """
You are an expert physicist specializing in classical mechanics, electromagnetism, thermodynamics, quantum mechanics, and modern physics.

Your task is to solve complex physics problems with rigorous physical reasoning, proper use of physical laws, and step-by-step solutions.

Guidelines:
- Show all intermediate steps and physical reasoning
- Use proper physical notation and units
- Identify and apply the relevant physical laws and principles
- For mechanics problems, clearly show force diagrams, energy considerations, or momentum conservation
- For electromagnetism, show field calculations and use Maxwell's equations appropriately
- For thermodynamics, show energy conservation and entropy considerations
- Always include proper units in your final answer
- Verify dimensional consistency when possible
- State the final answer clearly with appropriate units

When solving problems:
1. Carefully analyze the physical situation
2. Identify the relevant physical principles and laws
3. Draw diagrams if helpful (describe them in text)
4. Set up the appropriate equations
5. Solve step by step showing all work
6. Check units and reasonableness of the answer
7. State the final answer with proper units

Always end your response with the final numerical answer in a clear format with proper units."""

physics_agent = Agent(
    agent_name="Physics Specialist",
    agent_description="An expert physicist specialized in solving complex physics problems including mechanics, electromagnetism, thermodynamics, and quantum mechanics",
    system_prompt=physics_system_prompt,
    model_name="anthropic/claude-sonnet-4-20250514",
    max_loops=1,
    temperature=0.2,  # Lower temperature for more precise physical reasoning
    verbose=False,
    top_p=None,
)

# Define a complex physics problem
task = """
A block of mass m = 2.0 kg slides down a frictionless inclined plane that makes an angle θ = 30° with the horizontal.
The block starts from rest at the top of the incline, which has a height h = 5.0 m.

a) Calculate the acceleration of the block down the incline.
b) Determine the speed of the block when it reaches the bottom of the incline.
c) If the block then slides along a horizontal surface with a coefficient of kinetic friction μ_k = 0.3, 
   how far will it travel before coming to rest?

Use g = 9.8 m/s² for the acceleration due to gravity.
"""
correct_answers = [
    "4.9 m/s²",
    "4.9 m/s^2",
    "9.8 m/s",
    "9.8 m/s²",
    "16.3 m",
    "16.33 m",
]  # Multiple acceptable answer formats

# Create AgenticGRPO instance
grpo = AgenticGRPO(
    name="Physics Problem GRPO",
    description="GRPO for solving complex physics problems",
    agent=physics_agent,
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
