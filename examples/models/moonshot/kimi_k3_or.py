from dotenv import load_dotenv

from swarms import Agent

load_dotenv()

# Define an expanded system prompt tailored for customer support
system_prompt = (
    "You are Customer-Support-Agent, a highly skilled and empathetic virtual assistant dedicated to helping customers with a wide range of inquiries. "
    "Your primary goal is to resolve customer issues efficiently and courteously, providing clear, accurate, and friendly support at all times. "
    "Greet users warmly, actively listen to their concerns, and guide them through step-by-step solutions. "
    "If you need more information, politely request clarification. Summarize actions and offer next steps if needed. "
    "Always maintain a positive, professional tone, show patience, and reassure the user when appropriate. "
    "If a problem falls outside your scope, kindly recommend escalation or suggest alternative resources. "
    "Ensure all responses are easy to understand, avoid jargon, and prioritize customer satisfaction in every interaction."
)

# Initialize the agent for customer support
agent = Agent(
    agent_name="Customer-Support-Agent",
    agent_description="Empathetic and efficient AI agent for providing customer support across a variety of issues",
    system_prompt=system_prompt,
    model_name="openrouter/moonshotai/kimi-k3",
    max_loops=1,
    top_p=None,
    temperature=None,
    reasoning_effort="low",
    max_tokens=16000,
    tools_list_dictionary=None,
)

out = agent.run(
    task="A customer writes in: 'I'm having trouble logging into my account. Can you help me get back in?'",
)

print(out)
