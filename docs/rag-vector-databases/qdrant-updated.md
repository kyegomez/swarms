# Qdrant RAG Integration with Swarms

## Overview

Qdrant is a high-performance, open-source vector database designed specifically for similarity search and AI applications. It offers both cloud and self-hosted deployment options, providing enterprise-grade features including advanced filtering, payload support, collection aliasing, and clustering capabilities. Qdrant is built with Rust for optimal performance and memory efficiency, making it ideal for production RAG systems that require both speed and scalability.

## Key Features

- **High Performance**: Rust-based implementation with optimized HNSW algorithm
- **Rich Filtering**: Advanced payload filtering with complex query conditions
- **Collection Management**: Multiple collections with independent configurations
- **Payload Support**: Rich metadata storage and querying capabilities
- **Clustering**: Horizontal scaling with automatic sharding
- **Snapshots**: Point-in-time backups and recovery
- **Hybrid Cloud**: Flexible deployment options (cloud, on-premise, hybrid)
- **Real-time Updates**: Live index updates without service interruption

## Architecture

Qdrant integrates with Swarms agents as a flexible, high-performance vector database:

```
[Agent] -> [Qdrant Memory] -> [Vector Collections] -> [Similarity + Payload Search] -> [Retrieved Context]
```

The system leverages Qdrant's advanced filtering capabilities to combine vector similarity with rich metadata queries for precise context retrieval.

## Setup & Configuration

### Installation

```bash
pip install qdrant-client
pip install swarms
pip install litellm
```

### Environment Variables

```bash
# Qdrant Cloud configuration
export QDRANT_URL="https://your-cluster.qdrant.tech"
export QDRANT_API_KEY="your-api-key"

# Or local Qdrant configuration
export QDRANT_HOST="localhost"
export QDRANT_PORT="6333"

# OpenAI API key for LLM
export OPENAI_API_KEY="your-openai-api-key"
```

### Dependencies

- `qdrant-client>=1.7.0`
- `swarms`
- `litellm`
- `numpy`

## Code Example

```python
"""
Qdrant RAG Integration with Swarms Agent

This example demonstrates how to integrate Qdrant as a high-performance vector database
for RAG operations with Swarms agents using LiteLLM embeddings.
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from swarms import Agent
from litellm import embedding
import uuid
from datetime import datetime

class QdrantMemory:
    """Qdrant-based memory system for RAG operations"""
    
    def __init__(self, 
                 collection_name: str = "swarms_knowledge_base",
                 embedding_model: str = "text-embedding-3-small",
                 dimension: int = 1536,
                 distance_metric: Distance = Distance.COSINE,
                 hnsw_config: Optional[Dict] = None,
                 optimizers_config: Optional[Dict] = None):
        """
        Initialize Qdrant memory system
        
        Args:
            collection_name: Name of the Qdrant collection
            embedding_model: LiteLLM embedding model name  
            dimension: Vector dimension (1536 for text-embedding-3-small)
            distance_metric: Distance metric (COSINE, EUCLID, DOT)
            hnsw_config: HNSW algorithm configuration
            optimizers_config: Optimizer configuration for performance tuning
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.distance_metric = distance_metric
        
        # Default HNSW configuration optimized for typical use cases
        self.hnsw_config = hnsw_config or {
            "m": 16,                    # Number of bi-directional links
            "ef_construct": 200,        # Size of dynamic candidate list
            "full_scan_threshold": 10000, # When to use full scan vs HNSW
            "max_indexing_threads": 0,  # Auto-detect threads
            "on_disk": False           # Keep index in memory for speed
        }
        
        # Default optimizer configuration
        self.optimizers_config = optimizers_config or {
            "deleted_threshold": 0.2,     # Trigger optimization when 20% deleted
            "vacuum_min_vector_number": 1000, # Minimum vectors before vacuum
            "default_segment_number": 0,  # Auto-determine segments
            "max_segment_size": None,     # No segment size limit
            "memmap_threshold": None,     # Auto-determine memory mapping
            "indexing_threshold": 20000,  # Start indexing after 20k vectors
            "flush_interval_sec": 5,      # Flush interval in seconds
            "max_optimization_threads": 1 # Single optimization thread
        }
        
        # Initialize Qdrant client
        self.client = self._create_client()
        
        # Create collection if it doesn't exist
        self._create_collection()
        
    def _create_client(self) -> QdrantClient:
        """Create Qdrant client based on configuration"""
        # Try cloud configuration first
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        
        if url and api_key:
            print(f"Connecting to Qdrant Cloud: {url}")
            return QdrantClient(url=url, api_key=api_key)
        
        # Fall back to local configuration
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        
        print(f"Connecting to local Qdrant: {host}:{port}")
        return QdrantClient(host=host, port=port)
    
    def _create_collection(self):
        """Create collection with optimized configuration"""
        # Check if collection already exists
        collections = self.client.get_collections().collections
        if any(c.name == self.collection_name for c in collections):
            print(f"Collection '{self.collection_name}' already exists")
            return
        
        # Create collection with vector configuration
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.dimension,
                distance=self.distance_metric,
                hnsw_config=models.HnswConfigDiff(**self.hnsw_config)
            ),
            optimizers_config=models.OptimizersConfigDiff(**self.optimizers_config)
        )
        
        print(f"Created collection '{self.collection_name}' with {self.distance_metric} distance")
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using LiteLLM"""
        response = embedding(
            model=self.embedding_model,
            input=texts
        )
        return [item["embedding"] for item in response["data"]]
    
    def _generate_point_id(self) -> str:
        """Generate unique point ID"""
        return str(uuid.uuid4())
    
    def add_documents(self, 
                     documents: List[str], 
                     metadata: List[Dict] = None,
                     ids: List[str] = None,
                     batch_size: int = 100) -> List[str]:
        """Add multiple documents to Qdrant"""
        if metadata is None:
            metadata = [{}] * len(documents)
        
        if ids is None:
            ids = [self._generate_point_id() for _ in documents]
        
        # Generate embeddings
        embeddings = self._get_embeddings(documents)
        
        # Prepare points for upsert
        points = []
        for point_id, embedding_vec, doc, meta in zip(ids, embeddings, documents, metadata):
            # Create rich payload with document text and metadata
            payload = {
                "text": doc,
                "timestamp": datetime.now().isoformat(),
                **meta  # Include all metadata fields
            }
            
            points.append(PointStruct(
                id=point_id,
                vector=embedding_vec,
                payload=payload
            ))
        
        # Batch upsert points
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        print(f"Added {len(documents)} documents to Qdrant collection")
        return ids
    
    def add_document(self, 
                    document: str, 
                    metadata: Dict = None,
                    point_id: str = None) -> str:
        """Add a single document to Qdrant"""
        result = self.add_documents(
            documents=[document],
            metadata=[metadata or {}],
            ids=[point_id] if point_id else None
        )
        return result[0] if result else None
    
    def search(self, 
               query: str,
               limit: int = 3,
               score_threshold: float = None,
               payload_filter: Filter = None,
               with_payload: bool = True,
               with_vectors: bool = False) -> Dict[str, Any]:
        """Search for similar documents in Qdrant with advanced filtering"""
        
        # Generate query embedding
        query_embedding = self._get_embeddings([query])[0]
        
        # Perform search with optional filtering
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=payload_filter,
            with_payload=with_payload,
            with_vectors=with_vectors
        )
        
        # Format results
        formatted_results = {
            "documents": [],
            "metadata": [],
            "scores": [],
            "ids": []
        }
        
        for point in search_result:
            formatted_results["ids"].append(str(point.id))
            formatted_results["scores"].append(float(point.score))
            
            if with_payload and point.payload:
                # Extract text and separate metadata
                text = point.payload.get("text", "")
                # Remove internal fields from metadata
                metadata = {k: v for k, v in point.payload.items() 
                           if k not in ["text", "timestamp"]}
                
                formatted_results["documents"].append(text)
                formatted_results["metadata"].append(metadata)
            else:
                formatted_results["documents"].append("")
                formatted_results["metadata"].append({})
        
        return formatted_results
    
    def search_with_payload_filter(self, 
                                  query: str,
                                  filter_conditions: List[FieldCondition],
                                  limit: int = 3) -> Dict[str, Any]:
        """Search with complex payload filtering"""
        payload_filter = Filter(
            must=filter_conditions
        )
        
        return self.search(
            query=query,
            limit=limit,
            payload_filter=payload_filter
        )
    
    def delete_documents(self, 
                        point_ids: List[str] = None,
                        payload_filter: Filter = None) -> bool:
        """Delete documents from Qdrant"""
        if point_ids:
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=point_ids)
            )
        elif payload_filter:
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=payload_filter)
            )
        else:
            raise ValueError("Must specify either point_ids or payload_filter")
        
        return result.operation_id is not None
    
    def update_payload(self, 
                      point_ids: List[str],
                      payload: Dict,
                      overwrite: bool = False) -> bool:
        """Update payload for existing points"""
        operation = self.client.overwrite_payload if overwrite else self.client.set_payload
        
        result = operation(
            collection_name=self.collection_name,
            payload=payload,
            points=point_ids
        )
        
        return result.operation_id is not None
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get detailed collection information"""
        info = self.client.get_collection(self.collection_name)
        return {
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "segments_count": info.segments_count,
            "config": {
                "distance": info.config.params.vectors.distance,
                "dimension": info.config.params.vectors.size,
                "hnsw_config": info.config.params.vectors.hnsw_config.__dict__ if info.config.params.vectors.hnsw_config else None
            }
        }
    
    def create_payload_index(self, field_name: str, field_schema: str = "keyword"):
        """Create index on payload field for faster filtering"""
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name=field_name,
            field_schema=field_schema
        )
        print(f"Created payload index on field: {field_name}")
    
    def create_snapshot(self) -> str:
        """Create a snapshot of the collection"""
        result = self.client.create_snapshot(collection_name=self.collection_name)
        print(f"Created snapshot: {result.name}")
        return result.name

# Initialize Qdrant memory
memory = QdrantMemory(
    collection_name="swarms_rag_demo",
    embedding_model="text-embedding-3-small",
    dimension=1536,
    distance_metric=Distance.COSINE
)

# Sample documents for the knowledge base
documents = [
    "Qdrant is a high-performance, open-source vector database built with Rust for AI applications.",
    "RAG systems combine vector similarity search with rich payload filtering for precise context retrieval.",
    "Vector embeddings represent semantic meaning of text for similarity-based search operations.",
    "The Swarms framework integrates seamlessly with Qdrant's advanced filtering capabilities.",
    "LiteLLM provides unified access to embedding models with consistent API interfaces.",
    "HNSW algorithm in Qdrant provides excellent performance for approximate nearest neighbor search.",
    "Payload filtering enables complex queries combining vector similarity with metadata conditions.",
    "Qdrant supports both cloud and self-hosted deployment for flexible architecture options.",
]

# Rich metadata for advanced filtering demonstrations
metadatas = [
    {
        "category": "database", 
        "topic": "qdrant", 
        "difficulty": "intermediate",
        "type": "overview",
        "language": "rust",
        "performance_tier": "high"
    },
    {
        "category": "ai", 
        "topic": "rag", 
        "difficulty": "intermediate",
        "type": "concept",
        "language": "python",
        "performance_tier": "medium"
    },
    {
        "category": "ml", 
        "topic": "embeddings", 
        "difficulty": "beginner",
        "type": "concept",
        "language": "agnostic",
        "performance_tier": "medium"
    },
    {
        "category": "framework", 
        "topic": "swarms", 
        "difficulty": "beginner",
        "type": "integration",
        "language": "python",
        "performance_tier": "high"
    },
    {
        "category": "library", 
        "topic": "litellm", 
        "difficulty": "beginner",
        "type": "tool",
        "language": "python",
        "performance_tier": "medium"
    },
    {
        "category": "algorithm", 
        "topic": "hnsw", 
        "difficulty": "advanced",
        "type": "technical",
        "language": "rust",
        "performance_tier": "high"
    },
    {
        "category": "feature", 
        "topic": "filtering", 
        "difficulty": "advanced",
        "type": "capability",
        "language": "rust",
        "performance_tier": "high"
    },
    {
        "category": "deployment", 
        "topic": "architecture", 
        "difficulty": "intermediate",
        "type": "infrastructure",
        "language": "agnostic",
        "performance_tier": "high"
    }
]

# Add documents to Qdrant
print("Adding documents to Qdrant...")
doc_ids = memory.add_documents(documents, metadatas)
print(f"Successfully added {len(doc_ids)} documents")

# Create payload indices for better filtering performance
memory.create_payload_index("category")
memory.create_payload_index("difficulty")
memory.create_payload_index("performance_tier")

# Display collection information
info = memory.get_collection_info()
print(f"Collection info: {info['points_count']} points, {info['vectors_count']} vectors")

# Create Swarms agent with Qdrant RAG
agent = Agent(
    agent_name="Qdrant-RAG-Agent",
    agent_description="Advanced agent with Qdrant-powered RAG featuring rich payload filtering",
    model_name="gpt-4o",
    max_loops=1,
    dynamic_temperature_enabled=True,
)

def query_with_qdrant_rag(query_text: str, 
                         limit: int = 3, 
                         filter_conditions: List[FieldCondition] = None,
                         score_threshold: float = None):
    """Query with RAG using Qdrant's advanced filtering capabilities"""
    print(f"\nQuerying: {query_text}")
    if filter_conditions:
        print(f"Applied filters: {len(filter_conditions)} conditions")
    
    # Retrieve relevant documents using Qdrant
    if filter_conditions:
        results = memory.search_with_payload_filter(
            query=query_text,
            filter_conditions=filter_conditions,
            limit=limit
        )
    else:
        results = memory.search(
            query=query_text,
            limit=limit,
            score_threshold=score_threshold
        )
    
    if not results["documents"]:
        print("No relevant documents found")
        return agent.run(query_text)
    
    # Prepare context from retrieved documents
    context = "\n".join([
        f"Document {i+1}: {doc}" 
        for i, doc in enumerate(results["documents"])
    ])
    
    # Display retrieved documents with rich metadata
    print("Retrieved documents:")
    for i, (doc, score, meta) in enumerate(zip(
        results["documents"], results["scores"], results["metadata"]
    )):
        print(f"  {i+1}. (Score: {score:.4f})")
        print(f"     Category: {meta.get('category', 'N/A')}, Difficulty: {meta.get('difficulty', 'N/A')}")
        print(f"     Language: {meta.get('language', 'N/A')}, Performance: {meta.get('performance_tier', 'N/A')}")
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
        "What is Qdrant and what are its key features?",
        "How do RAG systems work with vector databases?",
        "What are the benefits of HNSW algorithm?",
        "How does payload filtering enhance search capabilities?",
    ]
    
    print("=== Basic RAG Queries ===")
    for query in queries:
        response = query_with_qdrant_rag(query, limit=3)
        print(f"Answer: {response}\n")
        print("-" * 80)
    
    # Test advanced payload filtering
    print("\n=== Advanced Payload Filtering ===")
    
    # Query only high-performance, advanced topics
    high_perf_filter = [
        FieldCondition(key="performance_tier", match=MatchValue(value="high")),
        FieldCondition(key="difficulty", match=MatchValue(value="advanced"))
    ]
    response = query_with_qdrant_rag(
        "What are high-performance advanced features?",
        limit=3,
        filter_conditions=high_perf_filter
    )
    print(f"High-performance features: {response}\n")
    
    # Query beginner-friendly Python content
    python_beginner_filter = [
        FieldCondition(key="language", match=MatchValue(value="python")),
        FieldCondition(key="difficulty", match=MatchValue(value="beginner"))
    ]
    response = query_with_qdrant_rag(
        "What Python tools should beginners know about?",
        limit=2,
        filter_conditions=python_beginner_filter
    )
    print(f"Python beginner tools: {response}\n")
    
    # Query database and algorithm concepts
    db_algo_filter = [
        FieldCondition(
            key="category", 
            match=MatchValue(any_of=["database", "algorithm"])
        )
    ]
    response = query_with_qdrant_rag(
        "Explain database and algorithm concepts",
        limit=3,
        filter_conditions=db_algo_filter
    )
    print(f"Database and algorithm concepts: {response}\n")
    
    # Demonstrate score threshold filtering
    print("=== Score Threshold Filtering ===")
    response = query_with_qdrant_rag(
        "What is machine learning?",  # Query not closely related to our documents
        limit=5,
        score_threshold=0.7  # High threshold to filter low-relevance results
    )
    print(f"High-relevance only: {response}\n")
    
    # Demonstrate payload updates
    print("=== Payload Updates ===")
    if doc_ids:
        # Update payload for first document
        memory.update_payload(
            point_ids=[doc_ids[0]],
            payload={"updated": True, "version": "2.0", "priority": "high"}
        )
        print("Updated document payload")
    
    # Add new document with advanced metadata
    new_doc = "Qdrant clustering enables horizontal scaling with automatic data distribution across nodes."
    new_metadata = {
        "category": "scaling", 
        "topic": "clustering", 
        "difficulty": "expert",
        "type": "feature",
        "language": "rust",
        "performance_tier": "ultra_high",
        "version": "1.0"
    }
    new_id = memory.add_document(new_doc, new_metadata)
    
    # Query about clustering with expert-level filter
    expert_filter = [
        FieldCondition(key="difficulty", match=MatchValue(value="expert")),
        FieldCondition(key="category", match=MatchValue(value="scaling"))
    ]
    response = query_with_qdrant_rag(
        "How does clustering work for scaling?",
        filter_conditions=expert_filter
    )
    print(f"Expert clustering info: {response}\n")
    
    # Demonstrate complex filtering with multiple conditions
    print("=== Complex Multi-Condition Filtering ===")
    complex_filter = [
        FieldCondition(key="performance_tier", match=MatchValue(any_of=["high", "ultra_high"])),
        FieldCondition(key="type", match=MatchValue(any_of=["feature", "capability", "technical"])),
        FieldCondition(key="difficulty", match=MatchValue(except_of=["beginner"]))
    ]
    response = query_with_qdrant_rag(
        "What are the most advanced technical capabilities?",
        limit=4,
        filter_conditions=complex_filter
    )
    print(f"Advanced capabilities: {response}\n")
    
    # Create snapshot for backup
    print("=== Collection Management ===")
    snapshot_name = memory.create_snapshot()
    
    # Display final collection statistics
    final_info = memory.get_collection_info()
    print(f"Final collection info:")
    print(f"  Points: {final_info['points_count']}")
    print(f"  Vectors: {final_info['vectors_count']}")
    print(f"  Indexed vectors: {final_info['indexed_vectors_count']}")
    print(f"  Segments: {final_info['segments_count']}")
    
    # Example of cleanup (use with caution)
    # Delete test documents
    # test_filter = Filter(
    #     must=[FieldCondition(key="category", match=MatchValue(value="test"))]
    # )
    # memory.delete_documents(payload_filter=test_filter)
```

## Use Cases

### 1. Advanced RAG Systems
- **Scenario**: Complex document retrieval requiring metadata filtering
- **Benefits**: Rich payload queries, high performance, flexible filtering
- **Best For**: Enterprise search, legal document analysis, technical documentation

### 2. Multi-Modal AI Applications  
- **Scenario**: Applications combining text, images, and other data types
- **Benefits**: Flexible payload structure, multiple vector configurations
- **Best For**: Content management, media analysis, cross-modal search

### 3. Real-time Analytics Platforms
- **Scenario**: Applications requiring fast vector search with real-time updates
- **Benefits**: Live index updates, high throughput, clustering support
- **Best For**: Recommendation engines, fraud detection, real-time personalization

### 4. Hybrid Cloud Deployments
- **Scenario**: Organizations requiring flexible deployment options
- **Benefits**: Cloud and on-premise options, data sovereignty, custom configurations
- **Best For**: Government, healthcare, financial services with strict compliance needs

## Performance Characteristics

### Scaling and Performance
- **Horizontal Scaling**: Clustering support with automatic sharding
- **Vertical Scaling**: Optimized memory usage and CPU utilization
- **Query Performance**: Sub-millisecond search with HNSW indexing
- **Update Performance**: Real-time updates without index rebuilding
- **Storage Efficiency**: Configurable on-disk vs in-memory storage

### HNSW Configuration Impact

| Configuration | Use Case | Memory | Speed | Accuracy |
|---------------|----------|---------|-------|----------|
| **m=16, ef=200** | Balanced | Medium | Fast | High |
| **m=32, ef=400** | High accuracy | High | Medium | Very High |
| **m=8, ef=100** | Memory optimized | Low | Very Fast | Medium |
| **Custom** | Specific workload | Variable | Variable | Variable |

### Performance Optimization
```python
# High-performance configuration
memory = QdrantMemory(
    hnsw_config={
        "m": 32,                    # More connections for accuracy
        "ef_construct": 400,        # Higher construction quality
        "full_scan_threshold": 5000, # Lower threshold for small datasets
        "max_indexing_threads": 4,  # Utilize multiple cores
        "on_disk": False           # Keep in memory for speed
    },
    optimizers_config={
        "indexing_threshold": 10000, # Start indexing sooner
        "max_optimization_threads": 2 # More optimization threads
    }
)
```

## Cloud vs Self-Hosted Deployment

### Qdrant Cloud
```python
# Cloud deployment
memory = QdrantMemory(
    collection_name="production-kb",
    # Configured via environment variables:
    # QDRANT_URL and QDRANT_API_KEY
)
```

**Advantages:**
- Managed infrastructure with automatic scaling
- Built-in monitoring and alerting
- Global edge locations for low latency
- Automatic backups and disaster recovery
- Enterprise security and compliance

### Self-Hosted Qdrant
```python
# Self-hosted deployment
memory = QdrantMemory(
    collection_name="private-kb",
    # Configured via environment variables:
    # QDRANT_HOST and QDRANT_PORT
)
```

**Advantages:**
- Full control over infrastructure and data
- Custom hardware optimization
- No data transfer costs
- Compliance with data residency requirements
- Custom security configurations

## Advanced Features

### Collection Aliases
```python
# Create collection alias for zero-downtime updates
client.create_alias(
    create_alias=models.CreateAlias(
        collection_name="swarms_kb_v2",
        alias_name="production_kb"
    )
)

# Switch traffic to new version
client.update_aliases(
    change_aliases_operations=[
        models.CreateAlias(
            collection_name="swarms_kb_v3",
            alias_name="production_kb"
        )
    ]
)
```

### Advanced Filtering Examples
```python
# Complex boolean logic
complex_filter = Filter(
    must=[
        FieldCondition(key="category", match=MatchValue(value="ai")),
        Filter(
            should=[
                FieldCondition(key="difficulty", match=MatchValue(value="advanced")),
                FieldCondition(key="performance_tier", match=MatchValue(value="high"))
            ]
        )
    ],
    must_not=[
        FieldCondition(key="deprecated", match=MatchValue(value=True))
    ]
)

# Range queries
date_filter = Filter(
    must=[
        FieldCondition(
            key="created_date",
            range=models.Range(
                gte="2024-01-01",
                lt="2024-12-31"
            )
        )
    ]
)

# Geo-queries (if using geo payloads)
geo_filter = Filter(
    must=[
        FieldCondition(
            key="location",
            geo_radius=models.GeoRadius(
                center=models.GeoPoint(lat=40.7128, lon=-74.0060),
                radius=1000.0  # meters
            )
        )
    ]
)
```

### Batch Operations
```python
# Efficient batch processing for large datasets
def batch_process_documents(memory, documents, batch_size=1000):
    """Process large document collections efficiently"""
    total_processed = 0
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        doc_texts = [doc['text'] for doc in batch]
        metadata = [doc['metadata'] for doc in batch]
        
        doc_ids = memory.add_documents(
            documents=doc_texts,
            metadata=metadata,
            batch_size=batch_size
        )
        
        total_processed += len(doc_ids)
        print(f"Processed {total_processed}/{len(documents)} documents")
    
    return total_processed
```

### Clustering Configuration
```python
# Configure Qdrant cluster (self-hosted)
cluster_config = {
    "nodes": [
        {"host": "node1.example.com", "port": 6333},
        {"host": "node2.example.com", "port": 6333}, 
        {"host": "node3.example.com", "port": 6333}
    ],
    "replication_factor": 2,
    "write_consistency_factor": 1
}

# Create distributed collection
client.create_collection(
    collection_name="distributed_kb",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    shard_number=6,  # Distribute across cluster
    replication_factor=2  # Redundancy
)
```

## Monitoring and Maintenance

### Health Monitoring
```python
def monitor_collection_health(memory):
    """Monitor collection health and performance"""
    info = memory.get_collection_info()
    
    # Check indexing progress
    index_ratio = info['indexed_vectors_count'] / info['vectors_count']
    if index_ratio < 0.9:
        print(f"Warning: Only {index_ratio:.1%} vectors are indexed")
    
    # Check segment efficiency
    avg_points_per_segment = info['points_count'] / info['segments_count']
    if avg_points_per_segment < 1000:
        print(f"Info: Low points per segment ({avg_points_per_segment:.0f})")
    
    return {
        'health_score': index_ratio,
        'points_per_segment': avg_points_per_segment,
        'status': 'healthy' if index_ratio > 0.9 else 'degraded'
    }
```

### Performance Tuning
```python
# Optimize collection for specific workload
def optimize_collection(client, collection_name, workload_type='balanced'):
    """Optimize collection configuration for workload"""
    configs = {
        'read_heavy': {
            'indexing_threshold': 1000,
            'max_indexing_threads': 8,
            'flush_interval_sec': 10
        },
        'write_heavy': {
            'indexing_threshold': 50000,
            'max_indexing_threads': 2,
            'flush_interval_sec': 1
        },
        'balanced': {
            'indexing_threshold': 20000,
            'max_indexing_threads': 4,
            'flush_interval_sec': 5
        }
    }
    
    config = configs.get(workload_type, configs['balanced'])
    
    client.update_collection(
        collection_name=collection_name,
        optimizers_config=models.OptimizersConfigDiff(**config)
    )
```

## Best Practices

1. **Collection Design**: Plan collection schema and payload structure upfront
2. **Index Strategy**: Create payload indices on frequently filtered fields
3. **Batch Operations**: Use batch operations for better throughput
4. **Memory Management**: Configure HNSW parameters based on available resources
5. **Filtering Optimization**: Use indexed fields for complex filter conditions
6. **Snapshot Strategy**: Regular snapshots for data backup and recovery
7. **Monitoring**: Implement health monitoring and performance tracking
8. **Cluster Planning**: Design cluster topology for high availability
9. **Version Management**: Use collection aliases for zero-downtime updates
10. **Security**: Implement proper authentication and network security

## Troubleshooting

### Common Issues

1. **Slow Query Performance**
   - Check if payload indices exist for filter fields
   - Verify HNSW configuration is appropriate for dataset size
   - Monitor segment count and optimization status

2. **High Memory Usage**
   - Enable on-disk storage for vectors if needed
   - Reduce HNSW 'm' parameter for memory efficiency
   - Monitor segment sizes and trigger optimization

3. **Indexing Delays**
   - Adjust indexing threshold based on write patterns
   - Increase max_indexing_threads if CPU allows
   - Monitor indexing progress and segment health

4. **Connection Issues**
   - Verify network connectivity and firewall settings
   - Check API key permissions and rate limits
   - Implement connection pooling and retry logic

### Debugging Tools
```python
# Debug collection status
def debug_collection(client, collection_name):
    """Debug collection issues"""
    info = client.get_collection(collection_name)
    
    print(f"Collection: {collection_name}")
    print(f"Status: {info.status}")
    print(f"Vectors: {info.vectors_count} total, {info.indexed_vectors_count} indexed")
    print(f"Segments: {info.segments_count}")
    
    # Check for common issues
    if info.vectors_count > 0 and info.indexed_vectors_count == 0:
        print("Issue: No vectors are indexed. Check indexing configuration.")
    
    if info.segments_count > info.vectors_count / 1000:
        print("Issue: Too many segments. Consider running optimization.")
```

This comprehensive guide provides everything needed to integrate Qdrant with Swarms agents for advanced RAG applications with rich payload filtering using the unified LiteLLM embeddings approach.