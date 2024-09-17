import os

from dotenv import load_dotenv

# Import the OpenAIChat model and the Agent struct
from swarms import Agent, OpenAIChat
from swarms_memory import ChromaDB

# Load the environment variables
load_dotenv()

# Get the API key from the environment
api_key = os.environ.get("OPENAI_API_KEY")


# Initilaize the chromadb client
chromadb = ChromaDB(
    metric="cosine",
    output_dir="scp",
    docs_folder="artifacts",
)

# Initialize the language model
llm = OpenAIChat(
    temperature=0.5,
    openai_api_key=api_key,
    max_tokens=1000,
)

## Initialize the workflow
agent = Agent(
    llm=llm,
    name="Health and Wellness Blog",
    system_prompt="Generate a 10,000 word blog on health and wellness.",
    max_loops=4,
    autosave=True,
    dashboard=True,
    long_term_memory=[chromadb],
    memory_chunk_size=300,
)

# Run the workflow on a task
agent.run("Generate a 10,000 word blog on health and wellness.")
