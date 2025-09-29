# Weaviate Cloud RAG Integration with Swarms

## Overview

Weaviate Cloud is a fully managed vector database service offering enterprise-grade vector search capabilities with built-in AI integrations. It combines GraphQL APIs with vector search, automatic schema inference, and native ML model integrations. Weaviate Cloud excels in multi-modal search, semantic understanding, and complex relationship modeling, making it ideal for sophisticated RAG applications requiring both vector similarity and graph-like data relationships.

## Key Features

- **GraphQL API**: Flexible query language for complex data retrieval
- **Multi-modal Search**: Support for text, images, and other data types
- **Built-in Vectorization**: Automatic embedding generation with various models
- **Schema Flexibility**: Dynamic schema with automatic type inference
- **Hybrid Search**: Combine vector similarity with keyword search
- **Graph Relationships**: Model complex data relationships
- **Enterprise Security**: SOC 2 compliance with role-based access control
- **Global Distribution**: Multi-region deployment with low latency

## Architecture

Weaviate Cloud integrates with Swarms agents as an intelligent, multi-modal vector database:

```
[Agent] -> [Weaviate Cloud Memory] -> [GraphQL + Vector Search] -> [Multi-modal Results] -> [Retrieved Context]
```

The system leverages Weaviate's GraphQL interface and built-in AI capabilities for sophisticated semantic search and relationship queries.

## Setup & Configuration

### Installation

```bash
pip install weaviate-client
pip install swarms
pip install litellm
```

### Environment Variables

```bash
# Weaviate Cloud credentials
export WEAVIATE_URL="https://your-cluster.weaviate.network"
export WEAVIATE_API_KEY="your-api-key"

# Optional: OpenAI API key (for built-in vectorization)
export OPENAI_API_KEY="your-openai-api-key"

# Optional: Additional model API keys
export COHERE_API_KEY="your-cohere-key"
export HUGGINGFACE_API_KEY="your-hf-key"
```

### Dependencies

- `weaviate-client>=4.4.0`
- `swarms`
- `litellm`
- `numpy`

## Code Example

```python
"""
Agent with Weaviate Cloud RAG

This example demonstrates using Weaviate Cloud as a vector database for RAG operations,
allowing agents to store and retrieve documents from cloud-hosted Weaviate.
"""

import os
from swarms import Agent
from swarms_memory import WeaviateDB


# Get Weaviate Cloud credentials
weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_key = os.getenv("WEAVIATE_API_KEY")

if not weaviate_url or not weaviate_key:
    print("Missing Weaviate Cloud credentials!")
    print("Please set WEAVIATE_URL and WEAVIATE_API_KEY environment variables")
    exit(1)

# Create WeaviateDB wrapper for cloud RAG operations
rag_db = WeaviateDB(
    embedding_model="text-embedding-3-small",
    collection_name="swarms_cloud_knowledge",
    cluster_url=f"https://{weaviate_url}",
    auth_client_secret=weaviate_key,
    distance_metric="cosine",
)

# Add documents to the cloud knowledge base
documents = [
    "Weaviate Cloud Service provides managed vector database hosting with enterprise features.",
    "Cloud-hosted vector databases offer scalability, reliability, and managed infrastructure.",
    "RAG combines retrieval and generation for more accurate AI responses.",
    "The Swarms framework supports multiple cloud memory backends including Weaviate Cloud.",
    "Swarms is the first and most reliable multi-agent production-grade framework.",
    "Kye Gomez is Founder and CEO of Swarms Corporation."
]

print("Adding documents to Weaviate Cloud...")
for doc in documents:
    rag_db.add(doc)

# Create agent with cloud RAG capabilities
agent = Agent(
    agent_name="Weaviate-Cloud-RAG-Agent",
    agent_description="Swarms Agent with Weaviate Cloud-powered RAG for scalable knowledge retrieval",
    model_name="gpt-4o",
    max_loops=1,
    dynamic_temperature_enabled=True,
    long_term_memory=rag_db
)

print("Testing agent with cloud RAG...")

# Query with cloud RAG
response = agent.run("What is Weaviate Cloud and how does it relate to RAG? Who founded Swarms?")
print(response)
```

## Use Cases

### 1. Multi-Modal Knowledge Systems
- **Scenario**: Applications requiring search across text, images, and other media
- **Benefits**: Native multi-modal support, unified search interface
- **Best For**: Content management, media libraries, educational platforms

### 2. Complex Relationship Modeling
- **Scenario**: Knowledge graphs with interconnected entities and relationships
- **Benefits**: GraphQL queries, relationship traversal, graph analytics
- **Best For**: Enterprise knowledge bases, research databases, social networks

### 3. Flexible Schema Applications
- **Scenario**: Rapidly evolving data structures and content types
- **Benefits**: Dynamic schema inference, automatic property addition
- **Best For**: Startups, experimental platforms, content aggregation systems

### 4. Enterprise Search Platforms
- **Scenario**: Large-scale enterprise search with complex filtering requirements
- **Benefits**: Advanced filtering, role-based access, enterprise security
- **Best For**: Corporate intranets, document management, compliance systems

## Performance Characteristics

### Search Types Performance

| Search Type | Use Case | Speed | Flexibility | Accuracy |
|-------------|----------|-------|-------------|----------|
| **Vector** | Semantic similarity | Fast | Medium | High |
| **Hybrid** | Combined semantic + keyword | Medium | High | Very High |
| **GraphQL** | Complex relationships | Variable | Very High | Perfect |
| **Multi-modal** | Cross-media search | Medium | Very High | High |

### Scaling and Deployment
- **Serverless**: Automatic scaling based on query load
- **Global**: Multi-region deployment for low latency
- **Multi-tenant**: Namespace isolation and access control
- **Performance**: Sub-100ms queries with proper indexing

## Best Practices

1. **Schema Design**: Plan class structure and property types upfront
2. **Vectorization Strategy**: Choose between built-in and external embeddings
3. **Query Optimization**: Use appropriate search types for different use cases
4. **Filtering Strategy**: Create indexed properties for frequent filters
5. **Batch Operations**: Use batch import for large datasets
6. **Monitoring**: Implement query performance monitoring
7. **Security**: Configure proper authentication and authorization
8. **Multi-modal**: Leverage native multi-modal capabilities when applicable

This comprehensive guide provides the foundation for integrating Weaviate Cloud with Swarms agents for sophisticated, multi-modal RAG applications using both built-in and LiteLLM embeddings approaches.