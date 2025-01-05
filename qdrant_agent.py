import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from swarm_models import Anthropic

from swarms import Agent


class QdrantMemory:
    def __init__(
        self,
        collection_name: str = "agent_memories",
        vector_size: int = 1536,  # Default size for Claude embeddings
        url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """Initialize Qdrant memory system.
        
        Args:
            collection_name: Name of the Qdrant collection to use
            vector_size: Dimension of the embedding vectors
            url: Optional Qdrant server URL (defaults to local)
            api_key: Optional Qdrant API key for cloud deployment
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Initialize Qdrant client
        if url and api_key:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(":memory:")  # Local in-memory storage
            
        # Create collection if it doesn't exist
        self._create_collection()
        
    def _create_collection(self):
        """Create the Qdrant collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        exists = any(col.name == self.collection_name for col in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            
    def add(
        self,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict] = None
    ) -> str:
        """Add a memory to the store.
        
        Args:
            text: The text content of the memory
            embedding: Vector embedding of the text
            metadata: Optional metadata to store with the memory
            
        Returns:
            str: ID of the stored memory
        """
        if metadata is None:
            metadata = {}
            
        # Add timestamp and generate ID
        memory_id = str(uuid.uuid4())
        metadata.update({
            "timestamp": datetime.utcnow().isoformat(),
            "text": text
        })
        
        # Store the point
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=memory_id,
                    payload=metadata,
                    vector=embedding
                )
            ]
        )
        
        return memory_id
    
    def query(
        self,
        query_embedding: List[float],
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> List[Dict]:
        """Query memories based on vector similarity.
        
        Args:
            query_embedding: Vector embedding of the query
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of matching memories with their metadata
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=score_threshold
        )
        
        memories = []
        for res in results:
            memory = res.payload
            memory["similarity_score"] = res.score
            memories.append(memory)
            
        return memories
    
    def delete(self, memory_id: str):
        """Delete a specific memory by ID."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=[memory_id]
            )
        )
        
    def clear(self):
        """Clear all memories from the collection."""
        self.client.delete_collection(self.collection_name)
        self._create_collection()

# # Example usage
# if __name__ == "__main__":
#     # Initialize memory
#     memory = QdrantMemory()
    
#     # Example embedding (would normally come from an embedding model)
#     example_embedding = np.random.rand(1536).tolist()
    
#     # Add a memory
#     memory_id = memory.add(
#         text="Important financial analysis about startup equity.",
#         embedding=example_embedding,
#         metadata={"category": "finance", "importance": "high"}
#     )
    
#     # Query memories
#     results = memory.query(
#         query_embedding=example_embedding,
#         limit=5
#     )
    
#     print(f"Found {len(results)} relevant memories")
#     for result in results:
#         print(f"Memory: {result['text']}")
#         print(f"Similarity: {result['similarity_score']:.2f}")

# Initialize memory with optional cloud configuration
memory = QdrantMemory(
    url=os.getenv("QDRANT_URL"),  # Optional: For cloud deployment
    api_key=os.getenv("QDRANT_API_KEY")  # Optional: For cloud deployment
)

# Model
model = Anthropic(anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))

# Initialize the agent with Qdrant memory
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    system_prompt="Agent system prompt here",
    agent_description="Agent performs financial analysis.",
    llm=model,
    long_term_memory=memory,
)

# Run a query
agent.run("What are the components of a startup's stock incentive equity plan?")