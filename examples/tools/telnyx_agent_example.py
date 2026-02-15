"""Example: Using Telnyx communication tools with Swarms agents.

This example demonstrates how to create a Swarms agent that can send SMS
messages and make phone calls using the Telnyx API.

Prerequisites:
    1. Install dependencies:
        pip install swarms telnyx

    2. Set environment variables:
        export TELNYX_API_KEY='your-api-key'
        export TELNYX_FROM_NUMBER='+15551234567'
        export OPENAI_API_KEY='your-openai-api-key'

Usage:
    python telnyx_agent_example.py
"""

from swarms import Agent
from swarms.tools.telnyx_tools import (
    telnyx_send_sms,
    telnyx_make_call,
    telnyx_hangup_call,
    telnyx_lookup_number,
)


# Create an agent with Telnyx communication tools
agent = Agent(
    agent_name="Communication-Agent",
    agent_description=(
        "An AI agent that can send SMS messages and make "
        "phone calls using the Telnyx telecommunications API"
    ),
    system_prompt=(
        "You are a helpful communication assistant. You can send "
        "SMS text messages and make phone calls using Telnyx. "
        "When asked to communicate with someone, use the "
        "appropriate tool. Always confirm the phone number is in "
        "E.164 format (e.g., +15551234567) before proceeding. "
        "After sending a message or making a call, report the "
        "result to the user."
    ),
    model_name="gpt-4o",
    max_loops=1,
    tools=[
        telnyx_send_sms,
        telnyx_make_call,
        telnyx_hangup_call,
        telnyx_lookup_number,
    ],
    dynamic_temperature_enabled=True,
)


# Example: Send an SMS
response = agent.run(
    "Send an SMS to +15551234567 with the message "
    "'Hello! This is a test message from a Swarms AI agent.'"
)
print(response)
