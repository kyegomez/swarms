"""
Marketplace Prompt Fetching Example

This example demonstrates loading prompts directly from the Swarms Marketplace
using the marketplace_prompt_id parameter. The agent automatically fetches
and uses the prompt as its system prompt.
"""

from swarms import Agent

agent = Agent(
    model_name="gpt-4.1",
    marketplace_prompt_id="6d165e47-1827-4abe-9a84-b25005d8e3b4",
    max_loops="auto",
    streaming_on=True,
    interactive=True,
)

response = agent.run("Hello, what can you help me with?")
print(response)
