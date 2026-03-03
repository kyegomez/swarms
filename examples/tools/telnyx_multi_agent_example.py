"""Example: Multi-agent customer outreach swarm with Telnyx.

This example shows how to use Telnyx tools in a multi-agent Swarms
setup where one agent researches prospects and another handles the
actual communication outreach.

Prerequisites:
    pip install swarms telnyx

    export TELNYX_API_KEY='your-api-key'
    export TELNYX_FROM_NUMBER='+15551234567'
    export OPENAI_API_KEY='your-openai-api-key'
"""

from swarms import Agent
from swarms.structs.sequential_workflow import SequentialWorkflow
from swarms.tools.telnyx_tools import (
    telnyx_send_sms,
    telnyx_lookup_number,
)


# Agent 1: Research agent that looks up phone numbers
research_agent = Agent(
    agent_name="Research-Agent",
    agent_description="Researches and validates phone numbers",
    system_prompt=(
        "You are a research assistant. When given a phone number, "
        "use the telnyx_lookup_number tool to validate it and "
        "gather carrier information. Report your findings clearly."
    ),
    model_name="gpt-4o",
    max_loops=1,
    tools=[telnyx_lookup_number],
)

# Agent 2: Communication agent that sends messages
comms_agent = Agent(
    agent_name="Communications-Agent",
    agent_description="Handles outbound SMS communications",
    system_prompt=(
        "You are a professional communications agent. Based on "
        "the research provided, compose and send appropriate SMS "
        "messages using the telnyx_send_sms tool. Keep messages "
        "professional, concise, and under 160 characters when "
        "possible."
    ),
    model_name="gpt-4o",
    max_loops=1,
    tools=[telnyx_send_sms],
)


# Create a sequential workflow: research first, then communicate
workflow = SequentialWorkflow(
    name="Customer-Outreach-Workflow",
    description="Research prospects and send outreach messages",
    agents=[research_agent, comms_agent],
)

# Run the workflow
result = workflow.run(
    "Look up the phone number +15551234567 to validate it, "
    "then send them a professional SMS introducing our AI "
    "services."
)

print(result)
