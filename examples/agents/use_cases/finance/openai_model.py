from swarms import OpenAIChat, Agent
import os

api_key = os.getenv("OPENAI_API_KEY")


# Create an instance of the OpenAIChat class
model = OpenAIChat(
    api_key=api_key,
    model_name="gpt-4o-mini",
    temperature=0.1,
    max_tokens=4000,
)

# Agent
agent = Agent(
    agent_name="Non-Profit Incorporation Agent",
    llm=model,
    system_prompt="I am an AI assistant that helps you incorporate a non-profit organization. I can provide information on the best states to incorporate a non-profit in, the steps to incorporate a non-profit, and answer any other questions you may have about non-profit incorporation.",
    max_loops="auto",
    interactive=True,
    streaming_on=True,
)


# Run
response = agent(
    "What's the best state to incorporate a non profit in?"
)
print(response)
