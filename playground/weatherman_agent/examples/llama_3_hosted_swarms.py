from swarms import llama3Hosted


# Example usage
llama3 = llama3Hosted(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    temperature=0.8,
    max_tokens=1000,
    system_prompt=(
        "You're a weather agent for Baron Weather, you specialize in"
        " weather analysis"
    ),
)

completion_generator = llama3.run(
    "What are the best weather conditions to lay concrete",
)

print(completion_generator)
