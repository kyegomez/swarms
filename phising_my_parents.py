import os

from swarms import OpenAIChat, Agent, AgentRearrange

# Purpose = To detect email spam using three different agents
agent1 = Agent(
    agent_name="SpamDetector1",
    system_prompt="Detect if the email is spam or not, and provide your reasoning",
    llm=OpenAIChat(openai_api_key=os.getenv("OPENAI_API_KEY")),
    max_loops=1,
    output_type=str,
    # tools=[],
    metadata="json",
    function_calling_format_type="OpenAI",
    function_calling_type="json",
    streaming_on=True,
)

agent2 = Agent(
    agent_name="SpamDetector2",
    system_prompt="Detect if the email is spam or not, and provide your reasoning",
    llm=OpenAIChat(openai_api_key=os.getenv("OPENAI_API_KEY")),
    max_loops=1,
    output_type=str,
    # tools=[],
    metadata="json",
    function_calling_format_type="OpenAI",
    function_calling_type="json",
    streaming_on=True,
)

agent3 = Agent(
    agent_name="SpamDetector3",
    system_prompt="Detect if the email is spam or not, and provide your reasoning",
    llm=OpenAIChat(openai_api_key=os.getenv("OPENAI_API_KEY")),
    max_loops=1,
    output_type=str,
    # tools=[],
    metadata="json",
    function_calling_format_type="OpenAI",
    function_calling_type="json",
    streaming_on=True,
)

swarm = AgentRearrange(
    flow="SpamDetector1 -> SpamDetector2 -> SpamDetector3",
    agents=[agent1, agent2, agent3],
    logging_enabled=True,
    max_loops=1,
)

# Run all the agents
swarm.run("Find YSL bag with the biggest discount")
