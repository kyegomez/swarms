# FAISS RAG Integration with Swarms

## Overview

FAISS (Facebook AI Similarity Search) is a library developed by Meta for efficient similarity search and clustering of dense vectors. It provides highly optimized algorithms for large-scale vector operations and is particularly well-suited for production RAG applications requiring high performance and scalability. FAISS excels at handling millions to billions of vectors with sub-linear search times.

## Key Features

- **High Performance**: Optimized C++ implementation with Python bindings
- **Multiple Index Types**: Support for various indexing algorithms (Flat, IVF, HNSW, PQ)
- **GPU Acceleration**: Optional GPU support for extreme performance
- **Memory Efficiency**: Compressed indexing options for large datasets
- **Exact and Approximate Search**: Configurable trade-offs between speed and accuracy
- **Batch Operations**: Efficient batch search and update operations
- **Clustering Support**: Built-in K-means clustering algorithms

## Architecture

FAISS integrates with Swarms agents as a high-performance vector store for RAG operations:

```
[Agent] -> [FAISS Memory] -> [Vector Index] -> [Similarity Search] -> [Retrieved Context]
```

The system stores document embeddings in optimized FAISS indices and performs ultra-fast similarity searches to provide relevant context to agents.

## Setup & Configuration

### Installation

```bash
pip install faiss-cpu  # CPU version
# OR
pip install faiss-gpu  # GPU version (requires CUDA)
pip install swarms
pip install litellm
pip install numpy
```

### Environment Variables

```bash
# OpenAI API key for LLM (if using OpenAI models)
export OPENAI_API_KEY="your-openai-api-key"

# Optional: Configure GPU usage
export CUDA_VISIBLE_DEVICES="0"
```

### Dependencies

- `faiss-cpu>=1.7.0` or `faiss-gpu>=1.7.0`
- `swarms`
- `litellm`
- `numpy`
- `pickle` (for persistence)

## Code Example

```python
"""
FAISS RAG Integration with Swarms Agent

This example demonstrates how to integrate FAISS as a high-performance vector database
for RAG operations with Swarms agents using LiteLLM embeddings.
"""

import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any
from swarms import Agent
from litellm import embedding

class FAISSMemory:
    """FAISS-based memory system for RAG operations"""
    
    def __init__(self, 
                 dimension: int = 1536,  # text-embedding-3-small dimension
                 embedding_model: str = "text-embedding-3-small",
                 index_type: str = "Flat",
                 nlist: int = 100):
        """
        Initialize FAISS memory system
        
        Args:
            dimension: Vector dimension (1536 for text-embedding-3-small)
            embedding_model: LiteLLM embedding model name
            index_type: FAISS index type ('Flat', 'IVF', 'HNSW')
            nlist: Number of clusters for IVF index
        """
        self.dimension = dimension
        self.embedding_model = embedding_model
        self.index_type = index_type
        self.nlist = nlist
        
        # Initialize FAISS index
        self.index = self._create_index()
        
        # Storage for documents and metadata
        self.documents = []
        self.metadata = []
        self.id_to_index = {}
        self.next_id = 0
        
    def _create_index(self):
        """Create appropriate FAISS index based on type"""
        if self.index_type == "Flat":
            # Exact search, good for small to medium datasets
            return faiss.IndexFlatIP(self.dimension)  # Inner product
            
        elif self.index_type == "IVF":
            # Approximate search with inverted file index
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
            return index
            
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World graphs
            index = faiss.IndexHNSWFlat(self.dimension, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 100
            return index
            
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using LiteLLM"""
        response = embedding(
            model=self.embedding_model,
            input=texts
        )
        embeddings = np.array([item["embedding"] for item in response["data"]])
        
        # Normalize for cosine similarity (convert to inner product)
        faiss.normalize_L2(embeddings)
        return embeddings.astype('float32')
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add multiple documents to the index"""
        if metadata is None:
            metadata = [{}] * len(documents)
            
        # Generate embeddings
        embeddings = self._get_embeddings(documents)
        
        # Train index if necessary (for IVF)
        if self.index_type == "IVF" and not self.index.is_trained:
            self.index.train(embeddings)
        
        # Add to index
        start_id = len(self.documents)
        self.index.add(embeddings)
        
        # Store documents and metadata
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        
        # Update ID mapping
        for i, doc in enumerate(documents):
            self.id_to_index[self.next_id] = start_id + i
            self.next_id += 1
            
        return list(range(start_id, start_id + len(documents)))
    
    def add_document(self, document: str, metadata: Dict = None):
        """Add a single document to the index"""
        return self.add_documents([document], [metadata or {}])[0]
    
    def search(self, query: str, k: int = 3, score_threshold: float = None) -> Dict[str, Any]:
        """Search for similar documents"""
        if len(self.documents) == 0:
            return {"documents": [], "metadata": [], "scores": [], "ids": []}
            
        # Generate query embedding
        query_embedding = self._get_embeddings([query])
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        scores = scores[0]  # Get first (and only) query result
        indices = indices[0]
        
        # Filter by score threshold if provided
        if score_threshold is not None:
            valid_indices = scores >= score_threshold
            scores = scores[valid_indices]
            indices = indices[valid_indices]
        
        # Prepare results
        results = {
            "documents": [self.documents[idx] for idx in indices if idx < len(self.documents)],
            "metadata": [self.metadata[idx] for idx in indices if idx < len(self.metadata)],
            "scores": scores.tolist(),
            "ids": indices.tolist()
        }
        
        return results
    
    def save_index(self, filepath: str):
        """Save FAISS index and metadata to disk"""
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.index")
        
        # Save metadata and documents
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'id_to_index': self.id_to_index,
                'next_id': self.next_id,
                'dimension': self.dimension,
                'embedding_model': self.embedding_model,
                'index_type': self.index_type,
                'nlist': self.nlist
            }, f)
    
    def load_index(self, filepath: str):
        """Load FAISS index and metadata from disk"""
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.index")
        
        # Load metadata and documents
        with open(f"{filepath}.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
            self.id_to_index = data['id_to_index']
            self.next_id = data['next_id']
            self.dimension = data['dimension']
            self.embedding_model = data['embedding_model']
            self.index_type = data['index_type']
            self.nlist = data.get('nlist', 100)

# Initialize FAISS memory system
# Option 1: Flat index (exact search, good for small datasets)
memory = FAISSMemory(
    dimension=1536,  # text-embedding-3-small dimension
    embedding_model="text-embedding-3-small",
    index_type="Flat"
)

# Option 2: IVF index (approximate search, good for large datasets)
# memory = FAISSMemory(
#     dimension=1536,
#     embedding_model="text-embedding-3-small",
#     index_type="IVF",
#     nlist=100
# )

# Option 3: HNSW index (very fast approximate search)
# memory = FAISSMemory(
#     dimension=1536,
#     embedding_model="text-embedding-3-small",
#     index_type="HNSW"
# )

# Sample documents for the knowledge base
documents = [
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "RAG combines retrieval and generation for more accurate AI responses with relevant context.",
    "Vector embeddings enable semantic search across large document collections.",
    "The Swarms framework supports multiple memory backends including FAISS for high performance.",
    "LiteLLM provides a unified interface for different embedding models and providers.",
    "FAISS supports both exact and approximate search algorithms for different use cases.",
    "GPU acceleration in FAISS can provide significant speedups for large-scale applications.",
    "Index types in FAISS include Flat, IVF, HNSW, and PQ for different performance characteristics.",
]

# Document metadata
metadatas = [
    {"category": "library", "topic": "faiss", "difficulty": "intermediate"},
    {"category": "ai", "topic": "rag", "difficulty": "intermediate"},
    {"category": "ai", "topic": "embeddings", "difficulty": "beginner"},
    {"category": "framework", "topic": "swarms", "difficulty": "beginner"},
    {"category": "library", "topic": "litellm", "difficulty": "beginner"},
    {"category": "search", "topic": "algorithms", "difficulty": "advanced"},
    {"category": "performance", "topic": "gpu", "difficulty": "advanced"},
    {"category": "indexing", "topic": "algorithms", "difficulty": "advanced"},
]

# Add documents to FAISS memory
print("Adding documents to FAISS index...")
doc_ids = memory.add_documents(documents, metadatas)
print(f"Added {len(doc_ids)} documents to the index")

# Create Swarms agent with FAISS-powered RAG
agent = Agent(
    agent_name="FAISS-RAG-Agent",
    agent_description="High-performance agent with FAISS-powered RAG for fast knowledge retrieval",
    model_name="gpt-4o",
    max_loops=1,
    dynamic_temperature_enabled=True,
)

def query_with_faiss_rag(query_text: str, k: int = 3):
    """Query with RAG using FAISS for high-performance retrieval"""
    print(f"\nQuerying: {query_text}")
    
    # Retrieve relevant documents using FAISS
    results = memory.search(query_text, k=k)
    
    if not results["documents"]:
        return agent.run(query_text)
    
    # Prepare context from retrieved documents
    context = "\n".join([
        f"Document {i+1}: {doc}" 
        for i, doc in enumerate(results["documents"])
    ])
    
    # Show retrieved documents and scores
    print("Retrieved documents:")
    for i, (doc, score) in enumerate(zip(results["documents"], results["scores"])):
        print(f"  {i+1}. (Score: {score:.4f}) {doc[:100]}...")
    
    # Enhanced prompt with context
    enhanced_prompt = f"""
Based on the following retrieved context, please answer the question:

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
    # Test different queries
    queries = [
        "What is FAISS and what are its key features?",
        "How does RAG work and why is it useful?",
        "What are the different FAISS index types?",
        "How can GPU acceleration improve performance?",
    ]
    
    for query in queries:
        response = query_with_faiss_rag(query, k=3)
        print(f"Answer: {response}\n")
        print("-" * 80)
    
    # Demonstrate adding new documents dynamically
    print("\nAdding new document...")
    new_doc = "FAISS supports product quantization (PQ) for memory-efficient storage of large vector datasets."
    new_metadata = {"category": "compression", "topic": "pq", "difficulty": "advanced"}
    memory.add_document(new_doc, new_metadata)
    
    # Query about the new document
    response = query_with_faiss_rag("What is product quantization in FAISS?")
    print(f"Answer about PQ: {response}")
    
    # Save the index for future use
    print("\nSaving FAISS index...")
    memory.save_index("./faiss_knowledge_base")
    print("Index saved successfully!")
    
    # Demonstrate loading (in a real application, you'd do this separately)
    print("\nTesting index loading...")
    new_memory = FAISSMemory()
    new_memory.load_index("./faiss_knowledge_base")
    test_results = new_memory.search("What is FAISS?", k=2)
    print(f"Loaded index test - found {len(test_results['documents'])} documents")
```

## Use Cases

### 1. Large-Scale Document Search
- **Scenario**: Searching through millions of documents or papers
- **Benefits**: Sub-linear search time, memory efficiency
- **Best For**: Academic databases, legal document search, news archives

### 2. Real-time Recommendation Systems
- **Scenario**: Product or content recommendations with low latency requirements
- **Benefits**: Ultra-fast query response, batch processing support
- **Best For**: E-commerce, streaming platforms, social media

### 3. High-Performance RAG Applications
- **Scenario**: Production RAG systems requiring fast response times
- **Benefits**: Optimized C++ implementation, GPU acceleration
- **Best For**: Customer support bots, technical documentation systems

### 4. Scientific Research Tools
- **Scenario**: Similarity search in scientific datasets or embeddings
- **Benefits**: Clustering support, exact and approximate search options
- **Best For**: Bioinformatics, materials science, computer vision research

## Performance Characteristics

### Index Types Performance Comparison

| Index Type | Search Speed | Memory Usage | Accuracy | Best Use Case |
|------------|--------------|--------------|----------|---------------|
| **Flat** | Fast | High | 100% | Small datasets (< 1M vectors) |
| **IVF** | Very Fast | Medium | 95-99% | Large datasets (1M-100M vectors) |
| **HNSW** | Ultra Fast | Medium-High | 95-98% | Real-time applications |
| **PQ** | Fast | Low | 90-95% | Memory-constrained environments |

### Scaling Characteristics
- **Small Scale** (< 1M vectors): Use Flat index for exact search
- **Medium Scale** (1M - 10M vectors): Use IVF with appropriate nlist
- **Large Scale** (10M - 1B vectors): Use IVF with PQ compression
- **Ultra Large Scale** (> 1B vectors): Use sharded indices across multiple machines

### Performance Optimization
```python
# GPU acceleration (if available)
import faiss
if faiss.get_num_gpus() > 0:
    gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index)

# Batch search for better throughput
results = memory.search_batch(queries, k=10)

# Memory mapping for very large indices
index = faiss.read_index("large_index.faiss", faiss.IO_FLAG_MMAP)
```

## Cloud vs Local Deployment

### Local Deployment
```python
# Local FAISS with persistence
memory = FAISSMemory(index_type="Flat")
memory.save_index("./local_faiss_index")
```

**Advantages:**
- No network latency
- Full control over hardware
- Cost-effective for development
- Easy debugging and profiling

**Disadvantages:**
- Limited by single machine resources
- Manual scaling required
- No built-in redundancy

### Cloud Deployment
```python
# Cloud deployment with distributed storage
# Use cloud storage for index persistence
import boto3
s3 = boto3.client('s3')

# Save to cloud storage
memory.save_index("/tmp/faiss_index")
s3.upload_file("/tmp/faiss_index.index", "bucket", "indices/faiss_index.index")
```

**Advantages:**
- Horizontal scaling with multiple instances
- Managed infrastructure
- Automatic backups and redundancy
- Global distribution

**Disadvantages:**
- Network latency for large indices
- Higher operational costs
- More complex deployment pipeline

## Advanced Configuration

### GPU Configuration
```python
import faiss

# Check GPU availability
print(f"GPUs available: {faiss.get_num_gpus()}")

# GPU-accelerated index
if faiss.get_num_gpus() > 0:
    cpu_index = faiss.IndexFlatIP(dimension)
    gpu_resources = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, cpu_index)
```

### Index Optimization
```python
# IVF index with optimized parameters
nlist = int(4 * np.sqrt(num_vectors))  # Rule of thumb
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
index.train(training_vectors)
index.nprobe = min(nlist, 10)  # Search parameter

# HNSW index optimization
index = faiss.IndexHNSWFlat(dimension, 32)  # M=32 connections
index.hnsw.efConstruction = 200  # Build-time parameter
index.hnsw.efSearch = 100  # Query-time parameter
```

### Memory Management
```python
# Product Quantization for memory efficiency
m = 8  # Number of subquantizers
nbits = 8  # Bits per subquantizer
pq = faiss.IndexPQ(dimension, m, nbits)

# Composite index (IVF + PQ)
index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)
```

## Best Practices

1. **Index Selection**: Choose appropriate index type based on dataset size and latency requirements
2. **Memory Management**: Use product quantization for large datasets with memory constraints
3. **Batch Processing**: Process documents and queries in batches for better throughput
4. **Normalization**: Normalize embeddings for cosine similarity using inner product indices
5. **Training Data**: Use representative data for training IVF indices
6. **Parameter Tuning**: Optimize nlist, nprobe, and other parameters for your specific use case
7. **Monitoring**: Track index size, query latency, and memory usage in production
8. **Persistence**: Regularly save indices and implement proper backup strategies

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce batch sizes or use product quantization
   - Consider using memory mapping for large indices
   - Monitor system memory usage

2. **Slow Search Performance**
   - Check if IVF index is properly trained
   - Adjust nprobe parameter (higher = slower but more accurate)
   - Consider using GPU acceleration

3. **Low Search Accuracy**
   - Increase nlist for IVF indices
   - Adjust efSearch for HNSW indices  
   - Verify embedding normalization

4. **Index Loading Issues**
   - Check file permissions and disk space
   - Verify FAISS version compatibility
   - Ensure consistent data types (float32)

This comprehensive guide provides everything needed to integrate FAISS with Swarms agents for high-performance RAG applications using the unified LiteLLM embeddings approach.