# ChromaDB RAG Integration with Swarms

## Overview

ChromaDB is an open-source embedding database designed to make it easy to build AI applications with embeddings. It provides a simple, fast, and scalable solution for storing and retrieving vector embeddings. ChromaDB is particularly well-suited for RAG (Retrieval-Augmented Generation) applications where you need to store document embeddings and perform similarity searches to enhance AI agent responses with relevant context.

## Key Features

- **Simple API**: Easy-to-use Python API for storing and querying embeddings
- **Multiple Storage Backends**: Supports in-memory, persistent local storage, and client-server modes
- **Metadata Filtering**: Advanced filtering capabilities with metadata
- **Multiple Distance Metrics**: Cosine, L2, and IP distance functions
- **Built-in Embedding Functions**: Support for various embedding models
- **Collection Management**: Organize embeddings into logical collections
- **Auto-embedding**: Automatic text embedding generation

## Architecture

ChromaDB integrates with Swarms agents by serving as the long-term memory backend. The architecture follows this pattern:

```
[Agent] -> [ChromaDB Memory] -> [Vector Store] -> [Similarity Search] -> [Retrieved Context]
```

The agent queries ChromaDB when it needs relevant context, and ChromaDB returns the most similar documents based on vector similarity.

## Setup & Configuration

### Installation

```bash
pip install chromadb
pip install swarms
pip install litellm
```

### Environment Variables

```bash
# Optional: For remote ChromaDB server
export CHROMA_HOST="localhost"
export CHROMA_PORT="8000"

# OpenAI API key for LLM (if using OpenAI models)
export OPENAI_API_KEY="your-openai-api-key"
```

### Dependencies

- `chromadb>=0.4.0`
- `swarms`
- `litellm`
- `numpy`

## Code Example

```python
"""
ChromaDB RAG Integration with Swarms Agent

This example demonstrates how to integrate ChromaDB as a vector database
for RAG operations with Swarms agents using LiteLLM embeddings.
"""

import chromadb
from swarms import Agent
import os
from litellm import embedding

# Initialize ChromaDB client
# Option 1: In-memory (for testing/development)
# client = chromadb.Client()

# Option 2: Persistent local storage (recommended for development)
client = chromadb.PersistentClient(path="./chroma_db")

# Option 3: Remote ChromaDB server (for production)
# client = chromadb.HttpClient(
#     host=os.getenv("CHROMA_HOST", "localhost"),
#     port=os.getenv("CHROMA_PORT", "8000")
# )

# Create or get collection
collection_name = "swarms_knowledge_base"
collection = client.get_or_create_collection(
    name=collection_name,
    metadata={"description": "Knowledge base for Swarms agents"}
)

# Embedding function using LiteLLM
def get_embeddings(texts):
    """Generate embeddings using LiteLLM unified interface"""
    if isinstance(texts, str):
        texts = [texts]
    
    response = embedding(
        model="text-embedding-3-small",  # Using LiteLLM unified approach
        input=texts
    )
    return [item["embedding"] for item in response["data"]]

# Sample documents to add to the knowledge base
documents = [
    "ChromaDB is an open-source embedding database for AI applications.",
    "RAG combines retrieval and generation for enhanced AI responses.",
    "Vector embeddings enable semantic search across documents.",
    "The Swarms framework supports multiple memory backends including ChromaDB.",
    "LiteLLM provides a unified interface for different embedding models.",
]

# Document metadata
metadatas = [
    {"category": "database", "topic": "chromadb", "difficulty": "beginner"},
    {"category": "ai", "topic": "rag", "difficulty": "intermediate"},
    {"category": "ai", "topic": "embeddings", "difficulty": "intermediate"},
    {"category": "framework", "topic": "swarms", "difficulty": "beginner"},
    {"category": "library", "topic": "litellm", "difficulty": "beginner"},
]

# Generate embeddings and add documents
embeddings = get_embeddings(documents)
document_ids = [f"doc_{i}" for i in range(len(documents))]

# Add documents to ChromaDB
collection.add(
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas,
    ids=document_ids
)

# Custom memory class for ChromaDB integration
class ChromaDBMemory:
    def __init__(self, collection, embedding_model="text-embedding-3-small"):
        self.collection = collection
        self.embedding_model = embedding_model
    
    def add(self, text, metadata=None):
        """Add a document to ChromaDB"""
        doc_id = f"doc_{len(self.collection.get()['ids'])}"
        embedding = get_embeddings([text])[0]
        
        self.collection.add(
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata] if metadata else [{}],
            ids=[doc_id]
        )
        return doc_id
    
    def query(self, query_text, n_results=3, where=None):
        """Query ChromaDB for relevant documents"""
        query_embedding = get_embeddings([query_text])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        return {
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0]
        }

# Initialize ChromaDB memory
memory = ChromaDBMemory(collection)

# Create Swarms agent with ChromaDB memory
agent = Agent(
    agent_name="ChromaDB-RAG-Agent",
    agent_description="Agent with ChromaDB-powered RAG for enhanced knowledge retrieval",
    model_name="gpt-4o",
    max_loops=1,
    dynamic_temperature_enabled=True,
    # Note: Integrating custom memory requires implementing the memory interface
)

# Function to query with context
def query_with_rag(query_text):
    """Query with RAG using ChromaDB"""
    # Retrieve relevant documents
    results = memory.query(query_text, n_results=3)
    
    # Prepare context
    context = "\n".join(results['documents'])
    
    # Enhanced prompt with context
    enhanced_prompt = f"""
    Based on the following context, please answer the question:
    
    Context:
    {context}
    
    Question: {query_text}
    
    Please provide a comprehensive answer based on the context provided.
    """
    
    # Run agent with enhanced prompt
    response = agent.run(enhanced_prompt)
    return response

# Example usage
if __name__ == "__main__":
    # Query with RAG
    question = "What is ChromaDB and how does it work with RAG?"
    response = query_with_rag(question)
    print(f"Question: {question}")
    print(f"Answer: {response}")
    
    # Add new document dynamically
    new_doc = "ChromaDB supports advanced filtering with metadata queries."
    memory.add(new_doc, {"category": "feature", "topic": "filtering"})
    
    # Query with filtering
    filtered_results = memory.query(
        "How to filter results?",
        n_results=2,
        where={"category": "feature"}
    )
    print(f"Filtered results: {filtered_results['documents']}")
```

## Use Cases

### 1. Knowledge Base RAG
- **Scenario**: Building a knowledge base for customer support
- **Benefits**: Fast semantic search, automatic embedding generation
- **Best For**: Small to medium-sized document collections

### 2. Development Documentation
- **Scenario**: Creating searchable documentation for development teams
- **Benefits**: Easy setup, local persistence, version control friendly
- **Best For**: Technical documentation, API references

### 3. Content Recommendations
- **Scenario**: Recommending relevant content based on user queries
- **Benefits**: Metadata filtering, multiple collections support
- **Best For**: Content management systems, educational platforms

### 4. Research Assistant
- **Scenario**: Building AI research assistants with paper databases
- **Benefits**: Complex metadata queries, collection organization
- **Best For**: Academic research, scientific literature review

## Performance Characteristics

### Scaling
- **Small Scale** (< 1M vectors): Excellent performance with in-memory storage
- **Medium Scale** (1M - 10M vectors): Good performance with persistent storage
- **Large Scale** (> 10M vectors): Consider distributed deployment or sharding

### Speed
- **Query Latency**: < 100ms for most queries
- **Insertion Speed**: ~1000 documents/second
- **Memory Usage**: Efficient with configurable caching

### Optimization Tips
1. **Batch Operations**: Use batch insert for better performance
2. **Metadata Indexing**: Design metadata schema for efficient filtering
3. **Collection Partitioning**: Use multiple collections for better organization
4. **Embedding Caching**: Cache embeddings for frequently accessed documents

## Cloud vs Local Deployment

### Local Deployment
```python
# In-memory (fastest, no persistence)
client = chromadb.Client()

# Persistent local (recommended for development)
client = chromadb.PersistentClient(path="./chroma_db")
```

**Advantages:**
- Fast development iteration
- No network latency
- Full control over data
- Cost-effective for small applications

**Disadvantages:**
- Limited scalability
- Single point of failure
- Manual backup required

### Cloud/Server Deployment
```python
# Remote ChromaDB server
client = chromadb.HttpClient(host="your-server.com", port=8000)
```

**Advantages:**
- Scalable architecture
- Centralized data management
- Professional backup solutions
- Multi-user access

**Disadvantages:**
- Network latency
- Additional infrastructure costs
- More complex deployment

## Configuration Options

### Distance Metrics
- **Cosine**: Best for normalized embeddings (default)
- **L2**: Euclidean distance for absolute similarity
- **IP**: Inner product for specific use cases

### Collection Settings
```python
collection = client.create_collection(
    name="my_collection",
    metadata={
        "hnsw:space": "cosine",  # Distance metric
        "hnsw:M": 16,           # HNSW graph connectivity
        "hnsw:ef_construction": 200,  # Build-time accuracy
        "hnsw:ef": 100          # Query-time accuracy
    }
)
```

### Memory Management
```python
# Configure client with memory limits
client = chromadb.PersistentClient(
    path="./chroma_db",
    settings={
        "anonymized_telemetry": False,
        "allow_reset": True,
        "persist_directory": "./chroma_storage"
    }
)
```

## Best Practices

1. **Collection Naming**: Use descriptive, consistent naming conventions
2. **Metadata Design**: Plan metadata schema for efficient filtering
3. **Batch Processing**: Use batch operations for better performance
4. **Error Handling**: Implement proper error handling and retry logic
5. **Monitoring**: Monitor collection sizes and query performance
6. **Backup Strategy**: Regular backups for persistent storage
7. **Version Management**: Track schema changes and migrations
8. **Security**: Implement proper authentication for production deployments

## Troubleshooting

### Common Issues

1. **Connection Errors**
   - Check ChromaDB server status
   - Verify network connectivity
   - Confirm correct host and port settings

2. **Performance Issues**
   - Monitor collection size and query complexity
   - Consider collection partitioning
   - Optimize metadata queries

3. **Memory Issues**
   - Adjust HNSW parameters
   - Use persistent storage instead of in-memory
   - Implement proper cleanup procedures

4. **Embedding Errors**
   - Verify LiteLLM configuration
   - Check API keys and quotas
   - Handle rate limiting properly

This comprehensive guide provides everything needed to integrate ChromaDB with Swarms agents for powerful RAG applications using the unified LiteLLM embeddings approach.