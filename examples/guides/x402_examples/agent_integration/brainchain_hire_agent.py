"""
Hire a model on BrainChain from a Swarms agent.

The agent pays 1 USDC on Base over x402 from its own wallet and gets
the work back once it passes grading. Work that is not accepted is
refunded to the paying wallet.

Install: pip install swarms usebrainchain

Set OPENAI_API_KEY, and BRAINCHAIN_PAYER_KEY to the private key of a
wallet that holds USDC on Base. Briefs and accepted results are
public, so keep secrets out of them.
"""

from swarms import Agent
from usebrainchain import hire_on_brainchain

agent = Agent(
    agent_name="BrainChain-Hiring-Agent",
    model_name="gpt-5.4",
    system_prompt=(
        "You hand self-contained tasks to a hired model with "
        "hire_on_brainchain. Write a short title, a brief that holds "
        "every fact the model needs, and three quality levels, best "
        "first. If the tool returns pending, call it again with the "
        "same arguments; that does not pay again. Report the accepted "
        "result, a refusal, or the pending status."
    ),
    tools=[hire_on_brainchain],
    max_loops=3,
)

print(
    agent.run(
        "Hire a model on BrainChain to write a Python function "
        "is_valid_isbn10(text) that returns True only for a valid "
        "ISBN-10, allowing hyphens and a final X."
    )
)
