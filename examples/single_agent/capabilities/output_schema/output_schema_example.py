from typing import List, Literal

from pydantic import BaseModel, Field

from swarms import Agent, AgentOutputSchemaError


class TicketTriage(BaseModel):
    category: Literal["billing", "bug", "feature_request", "other"]
    priority: Literal["low", "medium", "high"]
    summary: str = Field(
        description="One sentence about the problem."
    )
    next_steps: List[str]


agent = Agent(
    agent_name="Support-Triage-Agent",
    system_prompt="You triage customer support tickets.",
    model_name="gpt-5.4-mini",
    max_loops=1,
    output_schema=TicketTriage,
    # Send an invalid reply back to the model up to twice before raising.
    output_schema_retries=2,
)

ticket = (
    "I was charged twice for my March subscription and nobody has "
    "answered my emails in three days. Please refund the duplicate."
)

try:
    triage = agent.run(f"Triage this ticket:\n\n{ticket}")
except AgentOutputSchemaError as error:
    print(f"No valid TicketTriage: {error.validation_error}")
    print(error.raw_output)
else:
    # run returns a TicketTriage instance, not a string.
    print(type(triage).__name__, triage.category, triage.priority)
    print(triage.summary)
    for step in triage.next_steps:
        print(f"- {step}")
