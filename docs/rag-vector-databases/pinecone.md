# Pinecone RAG Integration with Swarms

## Overview

Pinecone is a fully managed vector database service designed specifically for high-performance AI applications. It provides a serverless, auto-scaling platform for vector similarity search that's optimized for production workloads. Pinecone offers enterprise-grade features including global distribution, real-time updates, metadata filtering, and comprehensive monitoring, making it ideal for production RAG systems that require reliability and scale.

## Key Features

- **Serverless Architecture**: Automatic scaling with pay-per-use pricing
- **Real-time Updates**: Live index updates without rebuilding
- **Global Distribution**: Multi-region deployment with low latency
- **Advanced Filtering**: Rich metadata filtering with complex queries
- **High Availability**: 99.9% uptime SLA with built-in redundancy
- **Performance Optimization**: Sub-millisecond query response times
- **Enterprise Security**: SOC 2 compliance with end-to-end encryption
- **Monitoring & Analytics**: Built-in observability and performance insights

## Architecture

Pinecone integrates with Swarms agents as a cloud-native vector database service:

```
[Agent] -> [Pinecone Memory] -> [Serverless Vector DB] -> [Global Search] -> [Retrieved Context]
```

The system leverages Pinecone's distributed infrastructure to provide consistent, high-performance vector operations across global regions.

## Setup & Configuration

### Installation

```bash
pip install pinecone-client
pip install swarms
pip install litellm
```

### Environment Variables

```bash
# Pinecone credentials
export PINECONE_API_KEY="your-pinecone-api-key"
export PINECONE_ENVIRONMENT="your-environment"  # e.g., "us-east1-gcp"

# Optional: Index configuration
export PINECONE_INDEX_NAME="swarms-knowledge-base"

# OpenAI API key for LLM
export OPENAI_API_KEY="your-openai-api-key"
```

### Dependencies

- `pinecone-client>=2.2.0`
- `swarms`
- `litellm`
- `numpy`

## Code Example

```python
"""
Pinecone RAG Integration with Swarms Agent

This example demonstrates how to integrate Pinecone as a serverless vector database
for RAG operations with Swarms agents using LiteLLM embeddings.
"""

import os
import time
from typing import List, Dict, Any, Optional
import numpy as np
import pinecone
from swarms import Agent
from litellm import embedding

class PineconeMemory:
    """Pinecone-based memory system for RAG operations"""
    
    def __init__(self, 
                 index_name: str = "swarms-knowledge-base",
                 embedding_model: str = "text-embedding-3-small",
                 dimension: int = 1536,
                 metric: str = "cosine",
                 pod_type: str = "p1.x1",
                 replicas: int = 1,
                 shards: int = 1):
        """
        Initialize Pinecone memory system
        
        Args:
            index_name: Name of the Pinecone index
            embedding_model: LiteLLM embedding model name  
            dimension: Vector dimension (1536 for text-embedding-3-small)
            metric: Distance metric (cosine, euclidean, dotproduct)
            pod_type: Pinecone pod type for performance/cost optimization
            replicas: Number of replicas for high availability
            shards: Number of shards for horizontal scaling
        """
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.metric = metric
        self.pod_type = pod_type
        self.replicas = replicas
        self.shards = shards
        
        # Initialize Pinecone connection
        self._initialize_pinecone()
        
        # Create or connect to index
        self.index = self._create_or_get_index()
        
        # Document counter for ID generation
        self._doc_counter = 0
        
    def _initialize_pinecone(self):
        """Initialize Pinecone with API credentials"""
        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT")
        
        if not api_key or not environment:
            raise ValueError("PINECONE_API_KEY and PINECONE_ENVIRONMENT must be set")
        
        pinecone.init(api_key=api_key, environment=environment)
        print(f"Initialized Pinecone in environment: {environment}")
        
    def _create_or_get_index(self):
        """Create or get the Pinecone index"""
        
        # Check if index exists
        if self.index_name in pinecone.list_indexes():
            print(f"Connecting to existing index: {self.index_name}")
            return pinecone.Index(self.index_name)
        
        # Create new index
        print(f"Creating new index: {self.index_name}")
        pinecone.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric=self.metric,
            pod_type=self.pod_type,
            replicas=self.replicas,
            shards=self.shards
        )
        
        # Wait for index to be ready
        print("Waiting for index to be ready...")
        while not pinecone.describe_index(self.index_name).status['ready']:
            time.sleep(1)
        
        print(f"Index {self.index_name} is ready!")
        return pinecone.Index(self.index_name)
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using LiteLLM"""
        response = embedding(
            model=self.embedding_model,
            input=texts
        )
        return [item["embedding"] for item in response["data"]]
    
    def _generate_id(self, prefix: str = "doc") -> str:
        """Generate unique document ID"""
        self._doc_counter += 1
        return f"{prefix}_{self._doc_counter}_{int(time.time())}"
    
    def add_documents(self, 
                     documents: List[str], 
                     metadata: List[Dict] = None,
                     ids: List[str] = None,
                     namespace: str = None,
                     batch_size: int = 100) -> List[str]:
        """Add multiple documents to Pinecone"""
        if metadata is None:
            metadata = [{}] * len(documents)
        
        if ids is None:
            ids = [self._generate_id() for _ in documents]
        
        # Generate embeddings
        embeddings = self._get_embeddings(documents)
        
        # Prepare vectors for upsert
        vectors = []
        for i, (doc_id, embedding_vec, doc, meta) in enumerate(
            zip(ids, embeddings, documents, metadata)
        ):
            # Add document text to metadata
            meta_with_text = {**meta, "text": doc}
            vectors.append({
                "id": doc_id,
                "values": embedding_vec,
                "metadata": meta_with_text
            })
        
        # Batch upsert vectors
        upserted_ids = []
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)
            upserted_ids.extend([v["id"] for v in batch])
        
        print(f"Added {len(documents)} documents to Pinecone index")
        return upserted_ids
    
    def add_document(self, 
                    document: str, 
                    metadata: Dict = None,
                    doc_id: str = None,
                    namespace: str = None) -> str:
        """Add a single document to Pinecone"""
        result = self.add_documents(
            documents=[document],
            metadata=[metadata or {}],
            ids=[doc_id] if doc_id else None,
            namespace=namespace
        )
        return result[0] if result else None
    
    def search(self, 
               query: str,
               top_k: int = 3,
               namespace: str = None,
               filter_dict: Dict = None,
               include_metadata: bool = True,
               include_values: bool = False) -> Dict[str, Any]:
        """Search for similar documents in Pinecone"""
        
        # Generate query embedding
        query_embedding = self._get_embeddings([query])[0]
        
        # Perform search
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            filter=filter_dict,
            include_metadata=include_metadata,
            include_values=include_values
        )
        
        # Format results
        formatted_results = {
            "documents": [],
            "metadata": [],
            "scores": [],
            "ids": []
        }
        
        for match in results.matches:
            formatted_results["ids"].append(match.id)
            formatted_results["scores"].append(float(match.score))
            
            if include_metadata and match.metadata:
                formatted_results["documents"].append(match.metadata.get("text", ""))
                # Remove text from metadata to avoid duplication
                meta_without_text = {k: v for k, v in match.metadata.items() if k != "text"}
                formatted_results["metadata"].append(meta_without_text)
            else:
                formatted_results["documents"].append("")
                formatted_results["metadata"].append({})
        
        return formatted_results
    
    def delete_documents(self, 
                        ids: List[str] = None,
                        filter_dict: Dict = None,
                        namespace: str = None,
                        delete_all: bool = False) -> Dict:
        """Delete documents from Pinecone"""
        if delete_all:
            return self.index.delete(delete_all=True, namespace=namespace)
        elif ids:
            return self.index.delete(ids=ids, namespace=namespace)
        elif filter_dict:
            return self.index.delete(filter=filter_dict, namespace=namespace)
        else:
            raise ValueError("Must specify ids, filter_dict, or delete_all=True")
    
    def get_index_stats(self, namespace: str = None) -> Dict:
        """Get index statistics"""
        return self.index.describe_index_stats().to_dict()
    
    def list_namespaces(self) -> List[str]:
        """List all namespaces in the index"""
        stats = self.index.describe_index_stats()
        return list(stats.namespaces.keys()) if stats.namespaces else []
    
    def update_document(self, 
                       doc_id: str,
                       document: str = None,
                       metadata: Dict = None,
                       namespace: str = None):
        """Update an existing document"""
        if document:
            # Generate new embedding if document text changed
            embedding_vec = self._get_embeddings([document])[0]
            metadata = metadata or {}
            metadata["text"] = document
            
            self.index.upsert(
                vectors=[{
                    "id": doc_id,
                    "values": embedding_vec,
                    "metadata": metadata
                }],
                namespace=namespace
            )
        elif metadata:
            # Update only metadata (requires fetching existing vector)
            fetch_result = self.index.fetch([doc_id], namespace=namespace)
            if doc_id in fetch_result.vectors:
                existing_vector = fetch_result.vectors[doc_id]
                updated_metadata = {**existing_vector.metadata, **metadata}
                
                self.index.upsert(
                    vectors=[{
                        "id": doc_id,
                        "values": existing_vector.values,
                        "metadata": updated_metadata
                    }],
                    namespace=namespace
                )

# Initialize Pinecone memory
memory = PineconeMemory(
    index_name="swarms-rag-demo",
    embedding_model="text-embedding-3-small",
    dimension=1536,
    metric="cosine",
    pod_type="p1.x1"  # Cost-effective for development
)

# Sample documents for the knowledge base
documents = [
    "Pinecone is a fully managed vector database service designed for AI applications at scale.",
    "RAG systems enhance AI responses by retrieving relevant context from knowledge bases.",
    "Vector embeddings enable semantic similarity search across unstructured data.",
    "The Swarms framework provides seamless integration with cloud vector databases like Pinecone.",
    "LiteLLM offers unified access to various embedding models through a consistent API.",
    "Serverless vector databases eliminate infrastructure management and provide auto-scaling.",
    "Real-time updates in Pinecone allow dynamic knowledge base modifications without downtime.",
    "Global distribution ensures low-latency access to vector search across worldwide regions.",
]

# Rich metadata for advanced filtering
metadatas = [
    {"category": "database", "topic": "pinecone", "difficulty": "beginner", "type": "overview", "industry": "tech"},
    {"category": "ai", "topic": "rag", "difficulty": "intermediate", "type": "concept", "industry": "ai"},
    {"category": "ml", "topic": "embeddings", "difficulty": "intermediate", "type": "concept", "industry": "ai"},
    {"category": "framework", "topic": "swarms", "difficulty": "beginner", "type": "integration", "industry": "ai"},
    {"category": "library", "topic": "litellm", "difficulty": "beginner", "type": "tool", "industry": "ai"},
    {"category": "architecture", "topic": "serverless", "difficulty": "advanced", "type": "concept", "industry": "cloud"},
    {"category": "feature", "topic": "realtime", "difficulty": "advanced", "type": "capability", "industry": "database"},
    {"category": "infrastructure", "topic": "global", "difficulty": "advanced", "type": "architecture", "industry": "cloud"},
]

# Add documents to Pinecone
print("Adding documents to Pinecone...")
doc_ids = memory.add_documents(documents, metadatas)
print(f"Successfully added {len(doc_ids)} documents")

# Display index statistics
stats = memory.get_index_stats()
print(f"Index stats: Total vectors: {stats.get('total_vector_count', 0)}")

# Create Swarms agent with Pinecone RAG
agent = Agent(
    agent_name="Pinecone-RAG-Agent",
    agent_description="Cloud-native agent with Pinecone-powered RAG for global-scale knowledge retrieval",
    model_name="gpt-4o",
    max_loops=1,
    dynamic_temperature_enabled=True,
)

def query_with_pinecone_rag(query_text: str, 
                           top_k: int = 3, 
                           filter_dict: Dict = None,
                           namespace: str = None):
    """Query with RAG using Pinecone for global-scale retrieval"""
    print(f"\nQuerying: {query_text}")
    if filter_dict:
        print(f"Filter: {filter_dict}")
    
    # Retrieve relevant documents using Pinecone
    results = memory.search(
        query=query_text,
        top_k=top_k,
        filter_dict=filter_dict,
        namespace=namespace
    )
    
    if not results["documents"]:
        print("No relevant documents found")
        return agent.run(query_text)
    
    # Prepare context from retrieved documents
    context = "\n".join([
        f"Document {i+1}: {doc}" 
        for i, doc in enumerate(results["documents"])
    ])
    
    # Display retrieved documents with metadata and scores
    print("Retrieved documents:")
    for i, (doc, score, meta) in enumerate(zip(
        results["documents"], results["scores"], results["metadata"]
    )):
        print(f"  {i+1}. (Score: {score:.4f}) Category: {meta.get('category', 'N/A')}")
        print(f"     Topic: {meta.get('topic', 'N/A')}, Industry: {meta.get('industry', 'N/A')}")
        print(f"     {doc[:100]}...")
    
    # Enhanced prompt with context
    enhanced_prompt = f"""
Based on the following retrieved context from our global knowledge base, please answer the question:

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
        "What is Pinecone and what makes it suitable for AI applications?",
        "How do RAG systems work and what are their benefits?",
        "What are the advantages of serverless vector databases?",
        "How does global distribution improve vector search performance?",
    ]
    
    print("=== Basic RAG Queries ===")
    for query in queries:
        response = query_with_pinecone_rag(query, top_k=3)
        print(f"Answer: {response}\n")
        print("-" * 80)
    
    # Test advanced filtering
    print("\n=== Advanced Filtering Queries ===")
    
    # Query only AI industry documents
    response = query_with_pinecone_rag(
        "What are key AI concepts?",
        top_k=3,
        filter_dict={"industry": "ai"}
    )
    print(f"AI concepts: {response}\n")
    
    # Query advanced topics in cloud/database industry
    response = query_with_pinecone_rag(
        "What are advanced cloud and database features?",
        top_k=2,
        filter_dict={
            "$and": [
                {"difficulty": "advanced"},
                {"$or": [{"industry": "cloud"}, {"industry": "database"}]}
            ]
        }
    )
    print(f"Advanced features: {response}\n")
    
    # Query concepts and overviews for beginners
    response = query_with_pinecone_rag(
        "What should beginners know about databases and frameworks?",
        top_k=3,
        filter_dict={
            "$and": [
                {"difficulty": "beginner"},
                {"$or": [{"category": "database"}, {"category": "framework"}]}
            ]
        }
    )
    print(f"Beginner content: {response}\n")
    
    # Demonstrate namespaces (optional)
    print("=== Namespace Example ===")
    # Add documents to a specific namespace
    namespace_docs = ["Pinecone supports namespaces for logical data separation and multi-tenancy."]
    namespace_meta = [{"category": "feature", "topic": "namespaces", "difficulty": "intermediate"}]
    memory.add_documents(namespace_docs, namespace_meta, namespace="features")
    
    # Query within namespace
    response = query_with_pinecone_rag(
        "How do namespaces work?",
        top_k=2,
        namespace="features"
    )
    print(f"Namespace query: {response}\n")
    
    # Demonstrate document update
    print("=== Document Update Example ===")
    # Update an existing document
    if doc_ids:
        memory.update_document(
            doc_id=doc_ids[0],
            metadata={"updated": True, "version": "2.0"}
        )
        print("Updated document metadata")
    
    # Add dynamic document
    new_doc = "Pinecone provides comprehensive monitoring and analytics for vector database operations."
    new_meta = {
        "category": "monitoring", 
        "topic": "analytics", 
        "difficulty": "intermediate",
        "industry": "database",
        "type": "feature"
    }
    new_id = memory.add_document(new_doc, new_meta)
    
    # Query about the new document
    response = query_with_pinecone_rag("What monitoring capabilities are available?")
    print(f"Monitoring capabilities: {response}\n")
    
    # Display final statistics
    final_stats = memory.get_index_stats()
    print(f"Final index stats: Total vectors: {final_stats.get('total_vector_count', 0)}")
    
    # List namespaces
    namespaces = memory.list_namespaces()
    print(f"Available namespaces: {namespaces}")
    
    # Example of cleanup (use with caution)
    # memory.delete_documents(filter_dict={"category": "test"})
```

## Use Cases

### 1. Production AI Applications
- **Scenario**: Customer-facing AI products requiring 99.9% uptime
- **Benefits**: Serverless scaling, global distribution, enterprise SLA
- **Best For**: SaaS platforms, mobile apps, web services

### 2. Real-time Recommendation Systems
- **Scenario**: E-commerce, content, or product recommendations
- **Benefits**: Sub-millisecond queries, real-time updates, global edge
- **Best For**: E-commerce platforms, streaming services, social media

### 3. Enterprise Knowledge Management
- **Scenario**: Large-scale corporate knowledge bases with global teams
- **Benefits**: Multi-region deployment, advanced security, comprehensive monitoring
- **Best For**: Fortune 500 companies, consulting firms, research organizations

### 4. Multi-tenant AI Platforms
- **Scenario**: AI platform providers serving multiple customers
- **Benefits**: Namespace isolation, flexible scaling, usage-based pricing
- **Best For**: AI service providers, B2B platforms, managed AI solutions

## Performance Characteristics

### Scaling
- **Serverless**: Automatic scaling based on traffic patterns
- **Global**: Multi-region deployment for worldwide low latency
- **Elastic**: Pay-per-use pricing model with no minimum commitments
- **High Availability**: 99.9% uptime SLA with built-in redundancy

### Performance Metrics
- **Query Latency**: < 10ms median, < 100ms 99th percentile
- **Throughput**: 10,000+ QPS per replica
- **Global Latency**: < 50ms from major worldwide regions
- **Update Latency**: Real-time updates with immediate consistency

### Pod Types and Performance

| Pod Type | Use Case | Performance | Cost | Best For |
|----------|----------|-------------|------|----------|
| **p1.x1** | Development, small apps | Good | Low | Prototypes, testing |
| **p1.x2** | Medium applications | Better | Medium | Production apps |
| **p1.x4** | High-performance apps | Best | High | Enterprise, high-traffic |
| **p2.x1** | Cost-optimized large scale | Good | Medium | Large datasets, batch processing |

## Cloud Deployment

### Production Configuration
```python
# High-performance production setup
memory = PineconeMemory(
    index_name="production-knowledge-base",
    embedding_model="text-embedding-3-small",
    pod_type="p1.x2",  # Higher performance
    replicas=2,         # High availability
    metric="cosine"
)
```

### Multi-region Setup
```python
# Configure for global deployment
import pinecone

# List available environments
environments = pinecone.list_environments()
print("Available regions:", environments)

# Choose optimal region based on user base
memory = PineconeMemory(
    index_name="global-knowledge-base",
    embedding_model="text-embedding-3-small",
    pod_type="p1.x2"
    # Environment set via PINECONE_ENVIRONMENT
)
```

### Cost Optimization
```python
# Cost-optimized configuration
memory = PineconeMemory(
    index_name="cost-optimized-kb",
    embedding_model="text-embedding-3-small",
    pod_type="p2.x1",  # Cost-optimized for large datasets
    replicas=1,        # Single replica for cost savings
    shards=1          # Single shard for simplicity
)
```

## Advanced Features

### Namespace Management
```python
# Organize data with namespaces
medical_docs = ["Medical knowledge documents..."]
legal_docs = ["Legal knowledge documents..."]

# Add to different namespaces
memory.add_documents(medical_docs, namespace="medical")
memory.add_documents(legal_docs, namespace="legal")

# Query specific namespace
medical_results = memory.search("medical query", namespace="medical")
legal_results = memory.search("legal query", namespace="legal")
```

### Complex Filtering
```python
# Advanced metadata filtering
complex_filter = {
    "$and": [
        {"category": {"$in": ["ai", "ml"]}},
        {"difficulty": {"$ne": "beginner"}},
        {"$or": [
            {"type": "concept"},
            {"type": "implementation"}
        ]}
    ]
}

results = memory.search(
    "advanced AI concepts",
    filter_dict=complex_filter,
    top_k=5
)
```

### Batch Operations
```python
# Efficient batch processing
large_dataset = load_large_document_collection()  # Your data loading logic

# Process in batches
batch_size = 100
for i in range(0, len(large_dataset), batch_size):
    batch = large_dataset[i:i + batch_size]
    documents = [item['text'] for item in batch]
    metadata = [item['metadata'] for item in batch]
    
    memory.add_documents(
        documents=documents,
        metadata=metadata,
        batch_size=batch_size
    )
```

### Real-time Updates
```python
# Dynamic knowledge base updates
def update_knowledge_base(new_documents, updated_documents, deleted_ids):
    """Update knowledge base in real-time"""
    # Add new documents
    if new_documents:
        memory.add_documents(new_documents)
    
    # Update existing documents
    for doc_id, content in updated_documents.items():
        memory.update_document(doc_id, content)
    
    # Remove outdated documents
    if deleted_ids:
        memory.delete_documents(ids=deleted_ids)
    
    print("Knowledge base updated in real-time")
```

## Monitoring and Analytics

### Built-in Metrics
```python
# Monitor index performance
stats = memory.get_index_stats()
print(f"Total vectors: {stats['total_vector_count']}")
print(f"Index fullness: {stats['index_fullness']}")

# Namespace statistics
for namespace, ns_stats in stats.get('namespaces', {}).items():
    print(f"Namespace '{namespace}': {ns_stats['vector_count']} vectors")
```

### Custom Monitoring
```python
import time
from datetime import datetime

class MonitoredPineconeMemory(PineconeMemory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_metrics = []
    
    def search(self, *args, **kwargs):
        start_time = time.time()
        results = super().search(*args, **kwargs)
        duration = time.time() - start_time
        
        # Log metrics
        self.query_metrics.append({
            'timestamp': datetime.now(),
            'duration': duration,
            'results_count': len(results['documents'])
        })
        
        return results
    
    def get_performance_stats(self):
        if not self.query_metrics:
            return {}
        
        durations = [m['duration'] for m in self.query_metrics]
        return {
            'avg_latency': sum(durations) / len(durations),
            'min_latency': min(durations),
            'max_latency': max(durations),
            'total_queries': len(self.query_metrics)
        }
```

## Best Practices

1. **Index Design**: Choose appropriate pod type based on performance requirements
2. **Metadata Strategy**: Design rich metadata schema for effective filtering
3. **Namespace Organization**: Use namespaces for logical data separation
4. **Batch Processing**: Use batch operations for better throughput and cost efficiency
5. **Error Handling**: Implement robust error handling with exponential backoff
6. **Monitoring**: Set up comprehensive monitoring and alerting
7. **Cost Management**: Monitor usage and optimize pod configuration
8. **Security**: Use API key rotation and access controls
9. **Regional Selection**: Choose regions closest to your users
10. **Version Management**: Track schema changes and implement migration strategies

## Troubleshooting

### Common Issues

1. **API Quota Exceeded**
   - Monitor API usage and implement rate limiting
   - Consider upgrading plan or optimizing query patterns
   - Use batch operations to reduce API calls

2. **High Latency**
   - Check pod type and consider upgrading
   - Verify regional configuration
   - Optimize query complexity and top_k values

3. **Index Capacity Issues**
   - Monitor index fullness metrics
   - Consider scaling up pod type or adding shards
   - Implement data archival strategies

4. **Connection Errors**
   - Verify API key and environment configuration
   - Check network connectivity and firewall settings
   - Implement retry logic with exponential backoff

### Performance Tuning
```python
# Optimize query performance
def optimized_search(memory, query, top_k=3):
    """Optimized search with caching and error handling"""
    try:
        results = memory.search(
            query=query,
            top_k=min(top_k, 10),  # Limit top_k for performance
            include_metadata=True,
            include_values=False   # Don't return vectors unless needed
        )
        return results
    except Exception as e:
        print(f"Search failed: {e}")
        # Implement fallback strategy
        return {"documents": [], "metadata": [], "scores": [], "ids": []}
```

This comprehensive guide provides everything needed to integrate Pinecone with Swarms agents for production-scale RAG applications using the unified LiteLLM embeddings approach.