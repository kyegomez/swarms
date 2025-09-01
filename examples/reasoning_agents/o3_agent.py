from swarms import Agent

# Initialize the LiteLLM wrapper with reasoning support
agent = Agent(
    model_name="claude-sonnet-4-20250514",  # OpenAI o3 model with reasoning
    reasoning_effort="low",  # Enable reasoning with high effort
    temperature=1,
    max_tokens=2000,
    stream=False,
    thinking_tokens=1024,
    top_p=0.95,
    streaming_on=True,
    print_on=False,
)


out = agent.run(
    task="Solve this step-by-step: A farmer has 17 sheep and all but 9 die. How many sheep does he have left?",
)

for chunk in out:
    # Flush
    print(chunk, end="", flush=True)
