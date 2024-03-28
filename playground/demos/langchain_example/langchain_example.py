import os

from langchain.llms import OpenAIChat

from swarms import Agent

# Initialize the model
llm = OpenAIChat(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=1000,
)

# Initialize the agent
agent = Agent(
    llm=llm,
    max_loops="auto",
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
)

# Run the workflow on a task
agent.run("Generate a 10,000 word blog on health and wellness.")
