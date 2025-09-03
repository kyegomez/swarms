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
Agent with ChromaDB RAG (Retrieval-Augmented Generation)

This example demonstrates using ChromaDB as a vector database for RAG operations,
allowing agents to store and retrieve documents for enhanced context.
"""

from swarms import Agent
from swarms_memory import ChromaDB


# Initialize ChromaDB wrapper for RAG operations
rag_db = ChromaDB(
    metric="cosine",  # Distance metric for similarity search
    output_dir="knowledge_base_new",  # Collection name
    limit_tokens=1000,  # Token limit for queries
    n_results=3,  # Number of results to retrieve
    verbose=False
)

# Add documents to the knowledge base
documents = [
    "ChromaDB is an open-source embedding database designed to store and query vector embeddings efficiently.",
    "ChromaDB provides a simple Python API for adding, querying, and managing vector embeddings with metadata.",
    "ChromaDB supports multiple embedding functions including OpenAI, Sentence Transformers, and custom models.",
    "ChromaDB can run locally or in distributed mode, making it suitable for both development and production.",
    "ChromaDB offers filtering capabilities allowing queries based on both vector similarity and metadata conditions.",
    "ChromaDB provides persistent storage and can handle large-scale embedding collections with fast retrieval.",
    "Kye Gomez is the founder of Swarms."
]

# Method 1: Add documents individually
for doc in documents:
    rag_db.add(doc)

# Create agent with RAG capabilities
agent = Agent(
    agent_name="RAG-Agent",
    agent_description="Swarms Agent with ChromaDB-powered RAG for enhanced knowledge retrieval",
    model_name="gpt-4o",
    max_loops=1,
    dynamic_temperature_enabled=True,
    long_term_memory=rag_db
)

# Query with RAG
response = agent.run("What is ChromaDB and who is founder of swarms ?")
print(response)
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

1. **Connection Errors**: Check ChromaDB server status. Verify network connectivity. Confirm correct host and port settings.

2. **Performance Issues**: Monitor collection size and query complexity. Consider collection partitioning. Optimize metadata queries.

3. **Memory Issues**: Adjust HNSW parameters. Use persistent storage instead of in-memory. Implement proper cleanup procedures.

4. **Embedding Errors**: Verify LiteLLM configuration. Check API keys and quotas. Handle rate limiting properly.

This comprehensive guide provides everything needed to integrate ChromaDB with Swarms agents for powerful RAG applications using the unified LiteLLM embeddings approach.