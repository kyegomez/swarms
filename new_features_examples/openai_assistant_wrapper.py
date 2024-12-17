from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)
from swarms.agents.openai_assistant import OpenAIAssistant

agent = OpenAIAssistant(
    name="test", instructions=FINANCIAL_AGENT_SYS_PROMPT
)

print(
    agent.run(
        "Create a table of super high growth opportunities for AI. I have $40k to invest in ETFs, index funds, and more. Please create a table in markdown.",
    )
)
