# Milvus Cloud RAG Integration with Swarms

## Overview

Milvus Cloud (also known as Zilliz Cloud) is a fully managed cloud service for Milvus, the world's most advanced open-source vector database. It provides enterprise-grade vector database capabilities with automatic scaling, high availability, and comprehensive security features. Milvus Cloud is designed for production-scale RAG applications that require robust performance, reliability, and minimal operational overhead.

## Key Features

- **Fully Managed Service**: No infrastructure management required
- **Auto-scaling**: Automatic scaling based on workload demands
- **High Availability**: Built-in redundancy and disaster recovery
- **Multiple Index Types**: Support for various indexing algorithms (IVF, HNSW, ANNOY, etc.)
- **Rich Metadata Filtering**: Advanced filtering capabilities with complex expressions
- **Multi-tenancy**: Secure isolation between different applications
- **Global Distribution**: Available in multiple cloud regions worldwide
- **Enterprise Security**: End-to-end encryption and compliance certifications

## Architecture

Milvus Cloud integrates with Swarms agents as a scalable, managed vector database solution:

```
[Agent] -> [Milvus Cloud Memory] -> [Managed Vector DB] -> [Similarity Search] -> [Retrieved Context]
```

The system leverages Milvus Cloud's distributed architecture to provide high-performance vector operations with enterprise-grade reliability.

## Setup & Configuration

### Installation

```bash
pip install pymilvus[cloud]
pip install swarms
pip install litellm
```

### Environment Variables

```bash
# Milvus Cloud credentials
export MILVUS_CLOUD_URI="https://your-cluster.api.milvuscloud.com"
export MILVUS_CLOUD_TOKEN="your-api-token"

# Optional: Database name (default: "default")
export MILVUS_DATABASE="your-database"

# OpenAI API key for LLM
export OPENAI_API_KEY="your-openai-api-key"
```

### Dependencies

- `pymilvus>=2.3.0`
- `swarms`
- `litellm`
- `numpy`

## Code Example

```python
"""
Milvus Cloud RAG Integration with Swarms Agent

This example demonstrates how to integrate Milvus Cloud as a managed vector database
for RAG operations with Swarms agents using LiteLLM embeddings.
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
from pymilvus import (
    connections, Collection, FieldSchema, CollectionSchema, 
    DataType, utility, MilvusClient
)
from swarms import Agent
from litellm import embedding

class MilvusCloudMemory:
    """Milvus Cloud-based memory system for RAG operations"""
    
    def __init__(self, 
                 collection_name: str = "swarms_knowledge_base",
                 embedding_model: str = "text-embedding-3-small",
                 dimension: int = 1536,
                 index_type: str = "HNSW",
                 metric_type: str = "COSINE"):
        """
        Initialize Milvus Cloud memory system
        
        Args:
            collection_name: Name of the Milvus collection
            embedding_model: LiteLLM embedding model name  
            dimension: Vector dimension (1536 for text-embedding-3-small)
            index_type: Index type (HNSW, IVF_FLAT, IVF_SQ8, etc.)
            metric_type: Distance metric (COSINE, L2, IP)
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.index_type = index_type
        self.metric_type = metric_type
        
        # Initialize Milvus Cloud connection
        self.client = self._connect_to_cloud()
        
        # Create collection if it doesn't exist
        self.collection = self._create_or_get_collection()
        
    def _connect_to_cloud(self):
        """Connect to Milvus Cloud using credentials"""
        uri = os.getenv("MILVUS_CLOUD_URI")
        token = os.getenv("MILVUS_CLOUD_TOKEN")
        
        if not uri or not token:
            raise ValueError("MILVUS_CLOUD_URI and MILVUS_CLOUD_TOKEN must be set")
        
        # Using MilvusClient for simplified operations
        client = MilvusClient(
            uri=uri,
            token=token
        )
        
        print(f"Connected to Milvus Cloud: {uri}")
        return client
        
    def _create_or_get_collection(self):
        """Create or get the collection with appropriate schema"""
        
        # Check if collection exists
        if self.client.has_collection(collection_name=self.collection_name):
            print(f"Collection '{self.collection_name}' already exists")
            return self.client.get_collection(self.collection_name)
        
        # Define collection schema
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_field=True
        )
        
        # Add fields
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.dimension)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        
        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema
        )
        
        # Create index on vector field
        index_params = {
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "params": self._get_index_params()
        }
        
        self.client.create_index(
            collection_name=self.collection_name,
            field_name="embedding",
            index_params=index_params
        )
        
        print(f"Created collection '{self.collection_name}' with {self.index_type} index")
        return self.client.get_collection(self.collection_name)
    
    def _get_index_params(self):
        """Get index parameters based on index type"""
        if self.index_type == "HNSW":
            return {"M": 16, "efConstruction": 200}
        elif self.index_type == "IVF_FLAT":
            return {"nlist": 128}
        elif self.index_type == "IVF_SQ8":
            return {"nlist": 128}
        elif self.index_type == "ANNOY":
            return {"n_trees": 8}
        else:
            return {}
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using LiteLLM"""
        response = embedding(
            model=self.embedding_model,
            input=texts
        )
        return [item["embedding"] for item in response["data"]]
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None) -> List[int]:
        """Add multiple documents to Milvus Cloud"""
        if metadata is None:
            metadata = [{}] * len(documents)
        
        # Generate embeddings
        embeddings = self._get_embeddings(documents)
        
        # Prepare data for insertion
        data = [
            {
                "embedding": emb,
                "text": doc,
                "metadata": meta
            }
            for emb, doc, meta in zip(embeddings, documents, metadata)
        ]
        
        # Insert data
        result = self.client.insert(
            collection_name=self.collection_name,
            data=data
        )
        
        # Flush to ensure data is written
        self.client.flush(collection_name=self.collection_name)
        
        print(f"Added {len(documents)} documents to Milvus Cloud")
        return result["ids"] if "ids" in result else []
    
    def add_document(self, document: str, metadata: Dict = None) -> int:
        """Add a single document to Milvus Cloud"""
        result = self.add_documents([document], [metadata or {}])
        return result[0] if result else None
    
    def search(self, 
               query: str, 
               limit: int = 3,
               filter_expr: str = None,
               output_fields: List[str] = None) -> Dict[str, Any]:
        """Search for similar documents in Milvus Cloud"""
        
        # Generate query embedding
        query_embedding = self._get_embeddings([query])[0]
        
        # Set default output fields
        if output_fields is None:
            output_fields = ["text", "metadata"]
        
        # Prepare search parameters
        search_params = {
            "metric_type": self.metric_type,
            "params": self._get_search_params()
        }
        
        # Perform search
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field="embedding",
            search_params=search_params,
            limit=limit,
            expr=filter_expr,
            output_fields=output_fields
        )[0]  # Get first (and only) query result
        
        # Format results
        formatted_results = {
            "documents": [],
            "metadata": [],
            "scores": [],
            "ids": []
        }
        
        for result in results:
            formatted_results["documents"].append(result.get("text", ""))
            formatted_results["metadata"].append(result.get("metadata", {}))
            formatted_results["scores"].append(float(result["distance"]))
            formatted_results["ids"].append(result["id"])
        
        return formatted_results
    
    def _get_search_params(self):
        """Get search parameters based on index type"""
        if self.index_type == "HNSW":
            return {"ef": 100}
        elif self.index_type in ["IVF_FLAT", "IVF_SQ8"]:
            return {"nprobe": 16}
        else:
            return {}
    
    def delete_documents(self, filter_expr: str) -> int:
        """Delete documents matching the filter expression"""
        result = self.client.delete(
            collection_name=self.collection_name,
            filter=filter_expr
        )
        print(f"Deleted documents matching: {filter_expr}")
        return result
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        stats = self.client.get_collection_stats(collection_name=self.collection_name)
        return {
            "row_count": stats["row_count"],
            "data_size": stats.get("data_size", "N/A"),
            "index_size": stats.get("index_size", "N/A")
        }

# Initialize Milvus Cloud memory
memory = MilvusCloudMemory(
    collection_name="swarms_rag_demo",
    embedding_model="text-embedding-3-small",
    dimension=1536,
    index_type="HNSW",  # High performance for similarity search
    metric_type="COSINE"
)

# Sample documents for the knowledge base
documents = [
    "Milvus Cloud is a fully managed vector database service with enterprise-grade features.",
    "RAG combines retrieval and generation to provide more accurate and contextual AI responses.",
    "Vector embeddings enable semantic search across unstructured data like text and images.",
    "The Swarms framework integrates with multiple vector databases including Milvus Cloud.",
    "LiteLLM provides a unified interface for different embedding models and providers.",
    "Milvus supports various index types including HNSW, IVF, and ANNOY for different use cases.",
    "Auto-scaling in Milvus Cloud ensures optimal performance without manual intervention.",
    "Enterprise security features include end-to-end encryption and compliance certifications.",
]

# Document metadata with rich attributes for filtering
metadatas = [
    {"category": "database", "topic": "milvus_cloud", "difficulty": "beginner", "type": "overview"},
    {"category": "ai", "topic": "rag", "difficulty": "intermediate", "type": "concept"},
    {"category": "ai", "topic": "embeddings", "difficulty": "intermediate", "type": "concept"},
    {"category": "framework", "topic": "swarms", "difficulty": "beginner", "type": "integration"},
    {"category": "library", "topic": "litellm", "difficulty": "beginner", "type": "tool"},
    {"category": "indexing", "topic": "algorithms", "difficulty": "advanced", "type": "technical"},
    {"category": "scaling", "topic": "cloud", "difficulty": "intermediate", "type": "feature"},
    {"category": "security", "topic": "enterprise", "difficulty": "advanced", "type": "feature"},
]

# Add documents to Milvus Cloud
print("Adding documents to Milvus Cloud...")
doc_ids = memory.add_documents(documents, metadatas)
print(f"Successfully added {len(doc_ids)} documents")

# Display collection statistics
stats = memory.get_collection_stats()
print(f"Collection stats: {stats}")

# Create Swarms agent with Milvus Cloud RAG
agent = Agent(
    agent_name="MilvusCloud-RAG-Agent",
    agent_description="Enterprise agent with Milvus Cloud-powered RAG for scalable knowledge retrieval",
    model_name="gpt-4o",
    max_loops=1,
    dynamic_temperature_enabled=True,
)

def query_with_milvus_rag(query_text: str, 
                         limit: int = 3, 
                         filter_expr: str = None):
    """Query with RAG using Milvus Cloud for enterprise-scale retrieval"""
    print(f"\nQuerying: {query_text}")
    if filter_expr:
        print(f"Filter: {filter_expr}")
    
    # Retrieve relevant documents using Milvus Cloud
    results = memory.search(
        query=query_text,
        limit=limit,
        filter_expr=filter_expr
    )
    
    if not results["documents"]:
        print("No relevant documents found")
        return agent.run(query_text)
    
    # Prepare context from retrieved documents
    context = "\n".join([
        f"Document {i+1}: {doc}" 
        for i, doc in enumerate(results["documents"])
    ])
    
    # Display retrieved documents with metadata
    print("Retrieved documents:")
    for i, (doc, score, meta) in enumerate(zip(
        results["documents"], results["scores"], results["metadata"]
    )):
        print(f"  {i+1}. (Score: {score:.4f}) Category: {meta.get('category', 'N/A')}")
        print(f"     {doc[:100]}...")
    
    # Enhanced prompt with context
    enhanced_prompt = f"""
Based on the following retrieved context from our knowledge base, please answer the question:

Context:
{context}

Question: {query_text}

Please provide a comprehensive answer based primarily on the context provided.
"""
    
    # Run agent with enhanced prompt
    response = agent.run(enhanced_prompt)
    return response

# Example usage and testing
if __name__ == "__main__":
    # Test basic queries
    queries = [
        "What is Milvus Cloud and what makes it enterprise-ready?",
        "How does RAG improve AI responses?",
        "What are the different index types supported by Milvus?",
        "What security features does Milvus Cloud provide?",
    ]
    
    print("=== Basic RAG Queries ===")
    for query in queries:
        response = query_with_milvus_rag(query, limit=3)
        print(f"Answer: {response}\n")
        print("-" * 80)
    
    # Test filtered queries using metadata
    print("\n=== Filtered Queries ===")
    
    # Query only advanced topics
    response = query_with_milvus_rag(
        "What are some advanced features?",
        limit=2,
        filter_expr='metadata["difficulty"] == "advanced"'
    )
    print(f"Advanced features: {response}\n")
    
    # Query only concepts
    response = query_with_milvus_rag(
        "Explain key AI concepts",
        limit=2, 
        filter_expr='metadata["type"] == "concept"'
    )
    print(f"AI concepts: {response}\n")
    
    # Query database-related documents
    response = query_with_milvus_rag(
        "Tell me about database capabilities",
        limit=3,
        filter_expr='metadata["category"] == "database" or metadata["category"] == "indexing"'
    )
    print(f"Database capabilities: {response}\n")
    
    # Demonstrate adding new documents with metadata
    print("=== Adding New Document ===")
    new_doc = "Milvus Cloud provides automatic backup and disaster recovery for enterprise data protection."
    new_metadata = {
        "category": "backup", 
        "topic": "disaster_recovery", 
        "difficulty": "intermediate",
        "type": "feature"
    }
    memory.add_document(new_doc, new_metadata)
    
    # Query about the new document
    response = query_with_milvus_rag("What backup features are available?")
    print(f"Backup features: {response}\n")
    
    # Display final collection statistics
    final_stats = memory.get_collection_stats()
    print(f"Final collection stats: {final_stats}")
    
    # Example of deleting documents (use with caution)
    # memory.delete_documents('metadata["category"] == "test"')
```

## Use Cases

### 1. Enterprise Knowledge Management
- **Scenario**: Large-scale corporate knowledge bases with millions of documents
- **Benefits**: Auto-scaling, high availability, enterprise security
- **Best For**: Fortune 500 companies, global organizations

### 2. Production RAG Applications
- **Scenario**: Customer-facing AI applications requiring 99.9% uptime
- **Benefits**: Managed infrastructure, automatic scaling, disaster recovery
- **Best For**: SaaS platforms, customer support systems

### 3. Multi-tenant Applications
- **Scenario**: Serving multiple customers with isolated data
- **Benefits**: Built-in multi-tenancy, secure data isolation
- **Best For**: AI platform providers, B2B SaaS solutions

### 4. Global AI Applications
- **Scenario**: Applications serving users worldwide
- **Benefits**: Global distribution, edge optimization
- **Best For**: International companies, global services

## Performance Characteristics

### Scaling
- **Auto-scaling**: Automatic compute and storage scaling based on workload
- **Horizontal Scaling**: Support for billions of vectors across multiple nodes
- **Vertical Scaling**: On-demand resource allocation for compute-intensive tasks

### Performance Metrics
- **Query Latency**: < 10ms for 95th percentile queries
- **Throughput**: 10,000+ QPS depending on configuration
- **Availability**: 99.9% uptime SLA
- **Consistency**: Tunable consistency levels

### Index Types Performance

| Index Type | Use Case | Performance | Memory | Accuracy |
|------------|----------|-------------|---------|----------|
| **HNSW** | High-performance similarity search | Ultra-fast | Medium | Very High |
| **IVF_FLAT** | Large datasets with exact results | Fast | High | Perfect |
| **IVF_SQ8** | Memory-efficient large datasets | Fast | Low | High |
| **ANNOY** | Read-heavy workloads | Very Fast | Low | High |

## Cloud vs Local Deployment

### Milvus Cloud Advantages
- **Fully Managed**: Zero infrastructure management
- **Enterprise Features**: Advanced security, compliance, monitoring
- **Global Scale**: Multi-region deployment capabilities
- **Cost Optimization**: Pay-per-use pricing model
- **Professional Support**: 24/7 technical support

### Configuration Options
```python
# Production configuration with advanced features
memory = MilvusCloudMemory(
    collection_name="production_knowledge_base",
    embedding_model="text-embedding-3-small",
    dimension=1536,
    index_type="HNSW",  # Best for similarity search
    metric_type="COSINE"
)

# Development configuration
memory = MilvusCloudMemory(
    collection_name="dev_knowledge_base",
    embedding_model="text-embedding-3-small", 
    dimension=1536,
    index_type="IVF_FLAT",  # Balanced performance
    metric_type="L2"
)
```

## Advanced Features

### Rich Metadata Filtering
```python
# Complex filter expressions
filter_expr = '''
(metadata["category"] == "ai" and metadata["difficulty"] == "advanced") 
or (metadata["topic"] == "embeddings" and metadata["type"] == "concept")
'''

results = memory.search(
    query="advanced AI concepts",
    limit=5,
    filter_expr=filter_expr
)
```

### Hybrid Search
```python
# Combine vector similarity with metadata filtering
results = memory.search(
    query="machine learning algorithms",
    limit=10,
    filter_expr='metadata["category"] in ["ai", "ml"] and metadata["difficulty"] != "beginner"'
)
```

### Collection Management
```python
# Create multiple collections for different domains
medical_memory = MilvusCloudMemory(
    collection_name="medical_knowledge",
    embedding_model="text-embedding-3-small"
)

legal_memory = MilvusCloudMemory(
    collection_name="legal_documents", 
    embedding_model="text-embedding-3-small"
)
```

## Best Practices

1. **Index Selection**: Choose HNSW for similarity search, IVF for large datasets
2. **Metadata Design**: Design rich metadata schema for effective filtering
3. **Batch Operations**: Use batch operations for better throughput
4. **Connection Pooling**: Implement connection pooling for production applications
5. **Error Handling**: Implement robust error handling and retry logic
6. **Monitoring**: Set up monitoring and alerting for performance metrics
7. **Cost Optimization**: Monitor usage and optimize collection configurations
8. **Security**: Follow security best practices for authentication and data access

## Monitoring and Observability

### Key Metrics to Monitor
- Query latency percentiles (p50, p95, p99)
- Query throughput (QPS)
- Error rates and types
- Collection size and growth
- Resource utilization

### Alerting Setup
```python
# Example monitoring integration
import logging

logger = logging.getLogger("milvus_rag")

def monitored_search(memory, query, **kwargs):
    start_time = time.time()
    try:
        results = memory.search(query, **kwargs)
        duration = time.time() - start_time
        logger.info(f"Search completed in {duration:.3f}s, found {len(results['documents'])} results")
        return results
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise
```

## Troubleshooting

### Common Issues

1. **Connection Errors**
   - Verify MILVUS_CLOUD_URI and MILVUS_CLOUD_TOKEN
   - Check network connectivity and firewall settings
   - Confirm cloud region accessibility

2. **Performance Issues**
   - Monitor collection size and index type appropriateness
   - Check query complexity and filter expressions
   - Review auto-scaling configuration

3. **Search Accuracy Issues**
   - Verify embedding model consistency
   - Check vector normalization if using cosine similarity
   - Review index parameters and search parameters

4. **Quota and Billing Issues**
   - Monitor usage against plan limits
   - Review auto-scaling settings
   - Check billing alerts and notifications

This comprehensive guide provides everything needed to integrate Milvus Cloud with Swarms agents for enterprise-scale RAG applications using the unified LiteLLM embeddings approach.