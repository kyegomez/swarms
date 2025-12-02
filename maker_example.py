import re

from swarms.structs.maker import MAKER


# Define task-specific functions for a counting task
def format_counting_prompt(
    task, state, step_idx, previous_result
):
    """Format prompt for counting task."""
    if previous_result is None:
        return f"{task}\nThis is step 1. What is the first number? Reply with just the number."
    return f"{task}\nThe previous number was {previous_result}. What is the next number? Reply with just the number."


def parse_counting_response(response):
    """Parse the counting response to extract the number."""
    numbers = re.findall(r"\d+", response)
    if numbers:
        return int(numbers[0])
    return response.strip()


def validate_counting_response(response, max_tokens):
    """Validate counting response."""
    if len(response) > max_tokens * 4:
        return False
    return bool(re.search(r"\d+", response))


# Create MAKER instance
maker = MAKER(
    name="CountingExample",
    description="MAKER example: counting numbers",
    model_name="gpt-4o-mini",
    system_prompt="You are a helpful assistant. When asked to count, respond with just the number, nothing else.",
    format_prompt=format_counting_prompt,
    parse_response=parse_counting_response,
    validate_response=validate_counting_response,
    k=2,
    max_tokens=100,
    temperature=0.1,
    verbose=True,
)

# Run the solver with the task as the main input
results = maker.run(
    task="Count from 1 to 10, one number at a time",
    max_steps=5,
)

print(results)

# Show statistics
stats = maker.get_statistics()
