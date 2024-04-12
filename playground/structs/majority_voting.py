from swarms import Agent, Anthropic, ChromaDB, MajorityVoting

# Initialize the llm
llm = Anthropic()

# Agents
agent1 = Agent(
    llm=llm,
    system_prompt=(
        "You are the leader of the Progressive Party. What is your"
        " stance on healthcare?"
    ),
    agent_name="Progressive Leader",
    agent_description="Leader of the Progressive Party",
    long_term_memory=ChromaDB(),
    max_steps=1,
)

agent2 = Agent(
    llm=llm,
    agent_name="Conservative Leader",
    agent_description="Leader of the Conservative Party",
    long_term_memory=ChromaDB(),
    max_steps=1,
)

agent3 = Agent(
    llm=llm,
    agent_name="Libertarian Leader",
    agent_description="Leader of the Libertarian Party",
    long_term_memory=ChromaDB(),
    max_steps=1,
)

# Initialize the majority voting
mv = MajorityVoting(
    agents=[agent1, agent2, agent3],
    output_parser=llm.majority_voting,
    autosave=False,
    verbose=True,
)


# Start the majority voting
mv.run("What is your stance on healthcare?")
