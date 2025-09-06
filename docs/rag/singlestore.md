# SingleStore RAG Integration with Swarms

## Overview

SingleStore is a distributed SQL database with native vector capabilities, combining the power of traditional relational operations with modern vector search functionality. It offers a unique approach to RAG by enabling complex queries that combine structured data, full-text search, and vector similarity in a single, high-performance system. SingleStore is ideal for applications requiring real-time analytics, complex data relationships, and high-throughput vector operations within a familiar SQL interface.

## Key Features

- **Unified SQL + Vector**: Combine relational queries with vector similarity search
- **Real-time Analytics**: Millisecond query performance on streaming data
- **Distributed Architecture**: Horizontal scaling across multiple nodes
- **HTAP Capabilities**: Hybrid transactional and analytical processing
- **Full-text Search**: Built-in text search with ranking and filtering
- **JSON Support**: Native JSON operations and indexing
- **High Throughput**: Handle millions of operations per second
- **Standard SQL**: Familiar SQL interface with vector extensions

## Architecture

SingleStore integrates with Swarms agents as a unified data platform combining vectors with structured data:

```
[Agent] -> [SingleStore Memory] -> [SQL + Vector Engine] -> [Hybrid Results] -> [Enriched Context]
```

The system enables complex queries combining vector similarity with traditional SQL operations for comprehensive data retrieval.

## Setup & Configuration

### Installation

```bash
pip install singlestoredb
pip install swarms
pip install litellm
```

### Environment Variables

```bash
# SingleStore connection
export SINGLESTORE_HOST="your-cluster.singlestore.com"
export SINGLESTORE_PORT="3306"
export SINGLESTORE_USER="your-username"
export SINGLESTORE_PASSWORD="your-password"
export SINGLESTORE_DATABASE="rag_database"

# Optional: SSL configuration
export SINGLESTORE_SSL_DISABLED="false"

# OpenAI API key for LLM
export OPENAI_API_KEY="your-openai-api-key"
```

### Dependencies

- `singlestoredb>=1.0.0`
- `swarms`
- `litellm`
- `numpy`
- `pandas` (for data manipulation)

## Code Example

```python
"""
Agent with SingleStore RAG (Retrieval-Augmented Generation)

This example demonstrates using SingleStore as a vector database for RAG operations,
allowing agents to store and retrieve documents for enhanced context.
"""

import os
from swarms import Agent
from swarms_memory import SingleStoreDB

# Initialize SingleStore wrapper for RAG operations
rag_db = SingleStoreDB(
    host=os.getenv("SINGLESTORE_HOST", "localhost"),
    port=int(os.getenv("SINGLESTORE_PORT", "3306")),
    user=os.getenv("SINGLESTORE_USER", "root"),
    password=os.getenv("SINGLESTORE_PASSWORD", "your-password"),
    database=os.getenv("SINGLESTORE_DATABASE", "knowledge_base"),
    table_name="documents",
    embedding_model="text-embedding-3-small"
)

# Add documents to the knowledge base
documents = [
    "SingleStore is a distributed SQL database designed for data-intensive applications.",
    "RAG combines retrieval and generation for more accurate AI responses.",
    "Vector embeddings enable semantic search across documents.",
    "The swarms framework supports multiple memory backends including SingleStore.",
    "Swarms is the first and most reliable multi-agent production-grade framework.",
    "Kye Gomez is Founder and CEO of Swarms."
]

# Add documents individually
for doc in documents:
    rag_db.add(doc)

# Create agent with RAG capabilities
agent = Agent(
    agent_name="RAG-Agent",
    agent_description="Swarms Agent with SingleStore-powered RAG for enhanced knowledge retrieval",
    model_name="gpt-4o",
    max_loops=1,
    dynamic_temperature_enabled=True,
    long_term_memory=rag_db
)

# Query with RAG
response = agent.run("What is SingleStore and how does it relate to RAG? Who is the founder of Swarms?")
print(response)
```

## Use Cases

### 1. **Enterprise Data Platforms**
- Combining operational data with knowledge bases
- Real-time analytics with contextual information
- Customer 360 views with vector similarity

### 2. **Financial Services**
- Risk analysis with document similarity
- Regulatory compliance with structured queries
- Fraud detection combining patterns and text

### 3. **E-commerce Platforms**
- Product recommendations with inventory data
- Customer support with order history
- Content personalization with user behavior

### 4. **Healthcare Systems**
- Patient records with research literature
- Drug discovery with clinical trial data
- Medical imaging with diagnostic text

## Performance Characteristics

### Query Performance
- **Vector Search**: < 10ms for millions of vectors
- **Hybrid Queries**: < 50ms combining SQL + vectors
- **Complex Joins**: Sub-second for structured + vector data
- **Real-time Ingestion**: 100K+ inserts per second

### Scaling Capabilities
- **Distributed**: Linear scaling across cluster nodes
- **Memory**: In-memory processing for hot data
- **Storage**: Tiered storage for cost optimization
- **Concurrency**: Thousands of concurrent queries

## Best Practices

1. **Schema Design**: Optimize table structure for query patterns
2. **Index Strategy**: Create appropriate indexes for filters and joins
3. **Vector Dimensions**: Choose optimal embedding dimensions for your use case
4. **Batch Processing**: Use batch operations for bulk data operations
5. **Query Optimization**: Leverage SQL query optimization techniques
6. **Memory Management**: Configure memory settings for optimal performance
7. **Monitoring**: Use SingleStore's built-in monitoring and metrics
8. **Security**: Implement proper authentication and access controls

This comprehensive guide provides everything needed to integrate SingleStore with Swarms agents for hybrid SQL + vector RAG applications, leveraging the power of unified data processing with the LiteLLM embeddings approach.