from swarms import Agent, OpenAIChat, MajorityVoting

api_key = os.getenv("OPENAI_API_KEY")
org_id = os.getenv("OPENAI_ORG_ID")
# Initialize the llm
llm = OpenAIChat(openai_api_key=api_key, openai_org_id=org_id, max_tokens=150)

# Initialize the agents
agent1 = Agent(llm=llm, max_loops=1)
agent2 = Agent(llm=llm, max_loops=1)
agent3 = Agent(llm=llm, max_loops=1)


# Initialize the majority voting
mv = MajorityVoting(
    agents=[agent1, agent2, agent3],
    concurrent=True,
    multithreaded=True,
)


# Start the majority voting
mv.run("What is the capital of France?")
