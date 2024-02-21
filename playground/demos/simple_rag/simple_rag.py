from swarms import Agent, ChromaDB, OpenAIChat

# Making an instance of the ChromaDB class
memory = ChromaDB(
    metric="cosine",
    n_results=3,
    output_dir="results",
    docs_folder="docs",
)

# Initializing the agent with the Gemini instance and other parameters
agent = Agent(
    agent_name="Covid-19-Chat",
    agent_description=(
        "This agent provides information about COVID-19 symptoms."
    ),
    llm=OpenAIChat(),
    max_loops="auto",
    autosave=True,
    verbose=True,
    long_term_memory=memory,
    stopping_condition="finish",
)

# Defining the task and image path
task = ("What are the symptoms of COVID-19?",)

# Running the agent with the specified task and image
out = agent.run(task)
print(out)
