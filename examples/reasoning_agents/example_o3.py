from swarms.utils.litellm_wrapper import LiteLLM

# Initialize the LiteLLM wrapper with reasoning support
llm = LiteLLM(
    model_name="claude-sonnet-4-20250514",  # OpenAI o3 model with reasoning
    reasoning_effort="low",  # Enable reasoning with high effort
    temperature=1,
    max_tokens=2000,
    stream=False,
    thinking_tokens=1024,
)

# Example task that would benefit from reasoning
task = "Solve this step-by-step: A farmer has 17 sheep and all but 9 die. How many sheep does he have left?"

print("=== Running reasoning model ===")
response = llm.run(task)
print(response)
