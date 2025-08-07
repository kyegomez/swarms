"""
Instructions:

1. Install the swarms package:
   > pip3 install -U swarms

2. Set the model name:
   > model_name = "openai/gpt-5-2025-08-07"

3. Add your OPENAI_API_KEY to the .env file and verify your account.

4. Run the agent!

Verify your OpenAI account here: https://platform.openai.com/settings/organization/general
"""

from swarms import Agent

agent = Agent(
    name="Research Agent",
    description="A research agent that can answer questions",
    model_name="openai/gpt-5-2025-08-07",
    streaming_on=True,
    max_loops=1,
    interactive=True,
)

out = agent.run(
    "What are the best arbitrage trading strategies for altcoins? Give me research papers and articles on the topic."
)

print(out)
