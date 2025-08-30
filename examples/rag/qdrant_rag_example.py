"""
Agent with Qdrant RAG (Retrieval-Augmented Generation)

This example demonstrates using Qdrant as a vector database for RAG operations,
allowing agents to store and retrieve documents for enhanced context.
"""

from qdrant_client import QdrantClient, models
from swarms import Agent
from swarms_memory import QdrantDB


# Initialize Qdrant client
# Option 1: In-memory (for testing/development - data is not persisted)
# client = QdrantClient(":memory:")

# Option 2: Local Qdrant server
# client = QdrantClient(host="localhost", port=6333)

# Option 3: Qdrant Cloud (recommended for production)
import os

client = QdrantClient(
    url=os.getenv("QDRANT_URL", "https://your-cluster.qdrant.io"),
    api_key=os.getenv("QDRANT_API_KEY", "your-api-key"),
)

# Create QdrantDB wrapper for RAG operations
rag_db = QdrantDB(
    client=client,
    embedding_model="text-embedding-3-small",
    collection_name="knowledge_base",
    distance=models.Distance.COSINE,
    n_results=3,
)

# Add documents to the knowledge base
documents = [
    "Qdrant is a vector database optimized for similarity search and AI applications.",
    "RAG combines retrieval and generation for more accurate AI responses.",
    "Vector embeddings enable semantic search across documents.",
    "The swarms framework supports multiple memory backends including Qdrant.",
]

# Method 1: Add documents individually
for doc in documents:
    rag_db.add(doc)

# Method 2: Batch add documents (more efficient for large datasets)
# Example with metadata
# documents_with_metadata = [
#     "Machine learning is a subset of artificial intelligence.",
#     "Deep learning uses neural networks with multiple layers.",
#     "Natural language processing enables computers to understand human language.",
#     "Computer vision allows machines to interpret visual information.",
#     "Reinforcement learning learns through interaction with an environment."
# ]
#
# metadata = [
#     {"category": "AI", "difficulty": "beginner", "topic": "overview"},
#     {"category": "ML", "difficulty": "intermediate", "topic": "neural_networks"},
#     {"category": "NLP", "difficulty": "intermediate", "topic": "language"},
#     {"category": "CV", "difficulty": "advanced", "topic": "vision"},
#     {"category": "RL", "difficulty": "advanced", "topic": "learning"}
# ]
#
# # Batch add with metadata
# doc_ids = rag_db.batch_add(documents_with_metadata, metadata=metadata, batch_size=3)
# print(f"Added {len(doc_ids)} documents in batch")
#
# # Query with metadata return
# results_with_metadata = rag_db.query(
#     "What is artificial intelligence?",
#     n_results=3,
#     return_metadata=True
# )
#
# for i, result in enumerate(results_with_metadata):
#     print(f"\nResult {i+1}:")
#     print(f"  Document: {result['document']}")
#     print(f"  Category: {result['category']}")
#     print(f"  Difficulty: {result['difficulty']}")
#     print(f"  Topic: {result['topic']}")
#     print(f"  Score: {result['score']:.4f}")

# Create agent with RAG capabilities
agent = Agent(
    agent_name="RAG-Agent",
    agent_description="Agent with Qdrant-powered RAG for enhanced knowledge retrieval",
    model_name="gpt-4o",
    max_loops=1,
    dynamic_temperature_enabled=True,
    long_term_memory=rag_db,
)

# Query with RAG
response = agent.run("What is Qdrant and how does it relate to RAG?")
print(response)
