from swarms.structs.agent import Agent
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
)

# Initialize the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create or get the index
index_name = "financial-agent-memory"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embedding_model.get_sentence_embedding_dimension(),
        metric="cosine",
        spec=ServerlessSpec(
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1"),
        ),
    )

# Get the index
pinecone_index = pc.Index(index_name)

# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    agent_description="Personal finance advisor agent",
    max_loops=4,
    model_name="gpt-5.4",
    dynamic_temperature_enabled=True,
    interactive=False,
    output_type="all",
)


def run_agent(task):
    # Run the agent and store the interaction
    result = agent.run(task)

    # Generate embedding for the document
    doc_text = f"Task: {task}\nResult: {result}"
    embedding = embedding_model.encode(doc_text).tolist()

    # Store the interaction in Pinecone
    pinecone_index.upsert(
        vectors=[
            {
                "id": str(datetime.now().timestamp()),
                "values": embedding,
                "metadata": {
                    "agent_name": agent.agent_name,
                    "task_type": "financial_analysis",
                    "timestamp": str(datetime.now()),
                    "text": doc_text,
                },
            }
        ]
    )

    return result


def query_memory(query_text, top_k=5):
    # Generate embedding for the query
    query_embedding = embedding_model.encode(query_text).tolist()

    # Query Pinecone
    results = pinecone_index.query(
        vector=query_embedding, top_k=top_k, include_metadata=True
    )

    return results


# print(out)
# print(type(out))
