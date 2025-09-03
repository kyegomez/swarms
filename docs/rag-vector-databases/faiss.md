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
Agent with FAISS RAG (Retrieval-Augmented Generation)

This example demonstrates using FAISS as a vector database for RAG operations,
allowing agents to store and retrieve documents for enhanced context.
"""

from swarms import Agent
from swarms_memory import FAISSDB

# Initialize FAISS wrapper for RAG operations
rag_db = FAISSDB(
    embedding_model="text-embedding-3-small",
    metric="cosine",
    index_file="knowledge_base.faiss"
)

# Add documents to the knowledge base
documents = [
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "RAG combines retrieval and generation for more accurate AI responses.",
    "Vector embeddings enable semantic search across documents.",
    "The swarms framework supports multiple memory backends including FAISS.",
    "Swarms is the first and most reliable multi-agent production-grade framework.",
    "Kye Gomez is Founder and CEO of Swarms."
]

# Add documents individually
for doc in documents:
    rag_db.add(doc)

# Create agent with RAG capabilities
agent = Agent(
    agent_name="RAG-Agent",
    agent_description="Swarms Agent with FAISS-powered RAG for enhanced knowledge retrieval",
    model_name="gpt-4o",
    max_loops=1,
    dynamic_temperature_enabled=True,
    long_term_memory=rag_db
)

# Query with RAG
response = agent.run("What is FAISS and how does it relate to RAG? Who is the founder of Swarms?")
print(response)
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