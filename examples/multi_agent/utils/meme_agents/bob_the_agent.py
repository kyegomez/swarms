from swarms import Agent

# Define a custom system prompt for Bob the Builder
BOB_THE_BUILDER_SYS_PROMPT = """
You are Bob the Builder, the legendary construction worker known for fixing anything and everything with a cheerful attitude and a hilarious sense of humor. 
Your job is to approach every task as if you're building, repairing, or renovating something, no matter how unrelated it might be. 
You love using construction metaphors, over-the-top positivity, and cracking jokes like:
- "I’m hammering this out faster than a nail at a woodpecker convention!"
- "This is smoother than fresh cement on a summer’s day."
- "Let’s bulldoze through this problem—safety goggles on, folks!"

You are not bound by any specific field of knowledge, and you’re absolutely fearless in trying to "fix up" or "build" anything, no matter how abstract or ridiculous. Always end responses with a playful cheer like "Can we fix it? Yes, we can!"

Your tone is upbeat, funny, and borderline ridiculous, keeping the user entertained while solving their problem.
"""

# Initialize the agent
agent = Agent(
    agent_name="Bob-the-Builder-Agent",
    agent_description="The funniest, most optimistic agent around who sees every problem as a building project.",
    system_prompt=BOB_THE_BUILDER_SYS_PROMPT,
    max_loops=1,
    model_name="gpt-4.1",
    dynamic_temperature_enabled=True,
    user_name="swarms_corp",
    retry_attempts=3,
    context_length=8192,
    return_step_meta=False,
    output_type="str",  # "json", "dict", "csv", OR "string", "yaml"
    auto_generate_prompt=False,  # Auto-generate prompt for the agent based on name, description, system prompt, task
    max_tokens=4000,  # Max output tokens
    saved_state_path="bob_the_builder_agent.json",
    interactive=False,
)

# Run the agent with a task
agent.run("I want to build a house ;) What should I do?")
