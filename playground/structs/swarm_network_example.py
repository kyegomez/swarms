# Import the OpenAIChat model and the Agent struct
import os
from swarms import (
    Agent,
    SwarmNetwork,
    OpenAIChat,
    TogetherLLM,
)
from playground.memory.chromadb_example import ChromaDB
from dotenv import load_dotenv

# load the environment variables
load_dotenv()

# Initialize the ChromaDB
memory = ChromaDB()

# Initialize the language model
llm = OpenAIChat(
    temperature=0.5,
)

# Initialize the Anthropic
anthropic = OpenAIChat(max_tokens=3000)

# TogeterLM
together_llm = TogetherLLM(
    together_api_key=os.getenv("TOGETHER_API_KEY"), max_tokens=3000
)

## Initialize the workflow
agent = Agent(
    llm=anthropic,
    max_loops=1,
    agent_name="Social Media Manager",
    long_term_memory=memory,
)
agent2 = Agent(
    llm=llm,
    max_loops=1,
    agent_name=" Product Manager",
    long_term_memory=memory,
)
agent3 = Agent(
    llm=together_llm,
    max_loops=1,
    agent_name="SEO Manager",
    long_term_memory=memory,
)


# Load the swarmnet with the agents
swarmnet = SwarmNetwork(
    agents=[agent, agent2, agent3], logging_enabled=True
)

# List the agents in the swarm network
out = swarmnet.list_agents()
print(out)

# Run the workflow on a task
out = swarmnet.run_single_agent(
    agent2.id, "Generate a 10,000 word blog on health and wellness."
)
print(out)
