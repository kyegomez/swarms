# Milvus Local/Lite RAG Integration with Swarms

## Overview

Milvus Lite is a lightweight, standalone version of Milvus that runs locally without requiring a full Milvus server deployment. It provides the core vector database functionality of Milvus in a simplified package that's perfect for development, testing, prototyping, and small-scale applications. Milvus Lite maintains compatibility with the full Milvus ecosystem while offering easier setup and deployment.

## Key Features

- **Zero Configuration**: No server setup or configuration required
- **Lightweight**: Minimal resource footprint for local development
- **Full Compatibility**: Same API as full Milvus for easy migration
- **Embedded Database**: Runs as a library within your application
- **Multiple Index Types**: Support for IVF, HNSW, and other algorithms
- **Persistent Storage**: Local file-based storage for data persistence
- **Python Native**: Pure Python implementation for easy installation
- **Cross-platform**: Works on Windows, macOS, and Linux

## Architecture

Milvus Lite integrates with Swarms agents as an embedded vector database solution:

```
[Agent] -> [Milvus Lite Memory] -> [Local Vector Store] -> [Similarity Search] -> [Retrieved Context]
```

The system runs entirely locally, providing fast vector operations without network overhead or external dependencies.

## Setup & Configuration

### Installation

```bash
pip install pymilvus[lite]  # Install with Milvus Lite support
pip install swarms
pip install litellm
```

### Environment Variables

```bash
# Optional: Specify database path
export MILVUS_LITE_DB_PATH="./milvus_lite.db"

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
Milvus Lite RAG Integration with Swarms Agent

This example demonstrates how to integrate Milvus Lite as a local vector database
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

class MilvusLiteMemory:
    """Milvus Lite-based memory system for RAG operations"""
    
    def __init__(self, 
                 db_path: str = "./milvus_lite.db",
                 collection_name: str = "swarms_knowledge_base",
                 embedding_model: str = "text-embedding-3-small",
                 dimension: int = 1536,
                 index_type: str = "HNSW",
                 metric_type: str = "COSINE"):
        """
        Initialize Milvus Lite memory system
        
        Args:
            db_path: Path to local Milvus Lite database file
            collection_name: Name of the Milvus collection
            embedding_model: LiteLLM embedding model name  
            dimension: Vector dimension (1536 for text-embedding-3-small)
            index_type: Index type (HNSW, IVF_FLAT, etc.)
            metric_type: Distance metric (COSINE, L2, IP)
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.index_type = index_type
        self.metric_type = metric_type
        
        # Initialize Milvus Lite connection
        self.client = self._connect_to_lite()
        
        # Create collection if it doesn't exist
        self.collection = self._create_or_get_collection()
        
    def _connect_to_lite(self):
        """Connect to Milvus Lite using local file"""
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        
        # Connect using MilvusClient with local file
        client = MilvusClient(uri=self.db_path)
        
        print(f"Connected to Milvus Lite database: {self.db_path}")
        return client
        
    def _create_or_get_collection(self):
        """Create or get the collection with appropriate schema"""
        
        # Check if collection exists
        if self.client.has_collection(collection_name=self.collection_name):
            print(f"Collection '{self.collection_name}' already exists")
            return self.collection_name
        
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
        return self.collection_name
    
    def _get_index_params(self):
        """Get index parameters based on index type"""
        if self.index_type == "HNSW":
            return {"M": 16, "efConstruction": 200}
        elif self.index_type == "IVF_FLAT":
            return {"nlist": 64}  # Smaller nlist for lite version
        elif self.index_type == "IVF_SQ8":
            return {"nlist": 64}
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
        """Add multiple documents to Milvus Lite"""
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
        
        print(f"Added {len(documents)} documents to Milvus Lite")
        return result.get("ids", [])
    
    def add_document(self, document: str, metadata: Dict = None) -> int:
        """Add a single document to Milvus Lite"""
        result = self.add_documents([document], [metadata or {}])
        return result[0] if result else None
    
    def search(self, 
               query: str, 
               limit: int = 3,
               filter_expr: str = None,
               output_fields: List[str] = None) -> Dict[str, Any]:
        """Search for similar documents in Milvus Lite"""
        
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
            return {"ef": 64}  # Lower ef for lite version
        elif self.index_type in ["IVF_FLAT", "IVF_SQ8"]:
            return {"nprobe": 8}  # Lower nprobe for lite version
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
            "data_size": stats.get("data_size", "N/A")
        }
    
    def backup_database(self, backup_path: str):
        """Create a backup of the Milvus Lite database"""
        import shutil
        shutil.copy2(self.db_path, backup_path)
        print(f"Database backed up to: {backup_path}")
    
    def get_database_size(self) -> int:
        """Get the size of the database file in bytes"""
        return os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

# Initialize Milvus Lite memory
memory = MilvusLiteMemory(
    db_path="./data/swarms_rag.db",
    collection_name="swarms_lite_demo",
    embedding_model="text-embedding-3-small",
    dimension=1536,
    index_type="HNSW",  # Efficient for local use
    metric_type="COSINE"
)

# Sample documents for the knowledge base
documents = [
    "Milvus Lite is a lightweight, standalone version of Milvus for local development and small applications.",
    "RAG systems combine document retrieval with text generation for more informed AI responses.",
    "Vector embeddings represent text as high-dimensional numerical vectors for semantic similarity.",
    "The Swarms framework provides flexible integration with various vector database backends.",
    "LiteLLM enables unified access to different embedding models through a single interface.",
    "Local vector databases like Milvus Lite eliminate network latency and external dependencies.",
    "HNSW indices provide excellent performance for similarity search in moderate-sized datasets.",
    "Embedded databases run within the application process for simplified deployment.",
]

# Document metadata for filtering and organization
metadatas = [
    {"category": "database", "topic": "milvus_lite", "difficulty": "beginner", "type": "overview"},
    {"category": "ai", "topic": "rag", "difficulty": "intermediate", "type": "concept"},
    {"category": "ml", "topic": "embeddings", "difficulty": "intermediate", "type": "concept"},
    {"category": "framework", "topic": "swarms", "difficulty": "beginner", "type": "integration"},
    {"category": "library", "topic": "litellm", "difficulty": "beginner", "type": "tool"},
    {"category": "performance", "topic": "local", "difficulty": "intermediate", "type": "benefit"},
    {"category": "indexing", "topic": "hnsw", "difficulty": "advanced", "type": "algorithm"},
    {"category": "architecture", "topic": "embedded", "difficulty": "intermediate", "type": "pattern"},
]

# Add documents to Milvus Lite
print("Adding documents to Milvus Lite...")
doc_ids = memory.add_documents(documents, metadatas)
print(f"Successfully added {len(doc_ids)} documents")

# Display database information
stats = memory.get_collection_stats()
db_size = memory.get_database_size()
print(f"Collection stats: {stats}")
print(f"Database size: {db_size / 1024:.1f} KB")

# Create Swarms agent with Milvus Lite RAG
agent = Agent(
    agent_name="MilvusLite-RAG-Agent",
    agent_description="Local agent with Milvus Lite-powered RAG for development and testing",
    model_name="gpt-4o",
    max_loops=1,
    dynamic_temperature_enabled=True,
)

def query_with_milvus_lite_rag(query_text: str, 
                              limit: int = 3, 
                              filter_expr: str = None):
    """Query with RAG using Milvus Lite for local, low-latency retrieval"""
    print(f"\nQuerying: {query_text}")
    if filter_expr:
        print(f"Filter: {filter_expr}")
    
    # Retrieve relevant documents using Milvus Lite
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
Based on the following retrieved context from our local knowledge base, please answer the question:

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
        "What is Milvus Lite and how is it different from full Milvus?",
        "How does RAG improve AI applications?",
        "What are the benefits of using local vector databases?",
        "How do HNSW indices work for similarity search?",
    ]
    
    print("=== Basic RAG Queries ===")
    for query in queries:
        response = query_with_milvus_lite_rag(query, limit=3)
        print(f"Answer: {response}\n")
        print("-" * 80)
    
    # Test filtered queries using metadata
    print("\n=== Filtered Queries ===")
    
    # Query only concepts
    response = query_with_milvus_lite_rag(
        "Explain key technical concepts",
        limit=2,
        filter_expr='metadata["type"] == "concept"'
    )
    print(f"Technical concepts: {response}\n")
    
    # Query beginner-level content
    response = query_with_milvus_lite_rag(
        "What should beginners know?",
        limit=3,
        filter_expr='metadata["difficulty"] == "beginner"'
    )
    print(f"Beginner content: {response}\n")
    
    # Query database-related documents
    response = query_with_milvus_lite_rag(
        "Tell me about database features",
        limit=2,
        filter_expr='metadata["category"] == "database" or metadata["category"] == "performance"'
    )
    print(f"Database features: {response}\n")
    
    # Demonstrate adding new documents dynamically
    print("=== Adding New Document ===")
    new_doc = "Milvus Lite supports persistent storage with automatic data recovery on restart."
    new_metadata = {
        "category": "persistence", 
        "topic": "storage", 
        "difficulty": "intermediate",
        "type": "feature"
    }
    memory.add_document(new_doc, new_metadata)
    
    # Query about the new document
    response = query_with_milvus_lite_rag("How does data persistence work?")
    print(f"Data persistence: {response}\n")
    
    # Demonstrate backup functionality
    print("=== Database Management ===")
    backup_path = "./data/swarms_rag_backup.db"
    memory.backup_database(backup_path)
    
    # Display final statistics
    final_stats = memory.get_collection_stats()
    final_db_size = memory.get_database_size()
    print(f"Final collection stats: {final_stats}")
    print(f"Final database size: {final_db_size / 1024:.1f} KB")
    
    # Example of cleaning up (optional)
    # memory.delete_documents('metadata["category"] == "test"')
```

## Use Cases

### 1. Local Development and Testing
- **Scenario**: Developing RAG applications without external dependencies
- **Benefits**: Zero setup, fast iteration, offline capability
- **Best For**: Prototype development, unit testing, local demos

### 2. Edge AI Applications
- **Scenario**: AI applications running on edge devices or offline environments
- **Benefits**: No internet required, low latency, privacy-first
- **Best For**: IoT devices, mobile apps, air-gapped systems

### 3. Desktop AI Applications
- **Scenario**: Personal AI assistants or productivity tools
- **Benefits**: Private data storage, instant startup, single-file deployment
- **Best For**: Personal knowledge management, desktop utilities

### 4. Small-Scale Production
- **Scenario**: Applications with limited data and users
- **Benefits**: Simple deployment, low resource usage, cost-effective
- **Best For**: MVPs, small businesses, specialized tools

## Performance Characteristics

### Resource Usage
- **Memory**: Low baseline usage (~50MB), scales with data size
- **Storage**: Efficient compression, typically 2-10x smaller than raw text
- **CPU**: Optimized algorithms, good performance on consumer hardware
- **Startup**: Fast initialization, typically < 1 second

### Scaling Limits
- **Vectors**: Recommended limit ~1M vectors for optimal performance
- **Memory**: Depends on available system RAM
- **Query Speed**: Sub-second response for most queries
- **Concurrent Access**: Single-process access (file locking)

### Performance Optimization
```python
# Optimize for small datasets
memory = MilvusLiteMemory(
    index_type="HNSW",
    metric_type="COSINE"
)

# Optimize for memory usage
memory = MilvusLiteMemory(
    index_type="IVF_FLAT",
    metric_type="L2"
)

# Batch operations for better performance
doc_ids = memory.add_documents(documents, metadata)
```

## Local vs Cloud Deployment

### Milvus Lite Advantages
- **No External Dependencies**: Runs completely offline
- **Privacy**: All data stays on local machine
- **Cost**: No cloud service fees
- **Simplicity**: Single file deployment
- **Development**: Fast iteration and debugging

### Limitations Compared to Full Milvus
- **Scalability**: Limited to single machine resources
- **Concurrency**: No multi-client support
- **Clustering**: No distributed deployment
- **Enterprise Features**: Limited monitoring and management tools

### Migration Path
```python
# Development with Milvus Lite
dev_memory = MilvusLiteMemory(
    db_path="./dev_database.db",
    collection_name="dev_collection"
)

# Production with full Milvus (same API)
# from pymilvus import connections
# connections.connect(host="prod-server", port="19530")
# prod_collection = Collection("prod_collection")
```

## File Management and Persistence

### Database Files
```python
# Default location
db_path = "./milvus_lite.db"

# Custom location with directory structure
db_path = "./data/vector_db/knowledge_base.db"

# Multiple databases for different domains
medical_memory = MilvusLiteMemory(db_path="./data/medical.db")
legal_memory = MilvusLiteMemory(db_path="./data/legal.db")
```

### Backup Strategies
```python
import shutil
import datetime

# Manual backup
backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
memory.backup_database(f"./backups/{backup_name}")

# Automated backup function
def create_scheduled_backup():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"./backups/auto_backup_{timestamp}.db"
    memory.backup_database(backup_path)
    return backup_path
```

### Data Migration
```python
# Export data for migration
def export_collection_data(memory):
    """Export all data from collection for migration"""
    # This would involve querying all documents and their metadata
    # Implementation depends on specific migration needs
    pass

# Import data from backup
def import_from_backup(source_path, target_memory):
    """Import data from another Milvus Lite database"""
    # Implementation for data transfer between databases
    pass
```

## Development Workflow

### Testing Setup
```python
import tempfile
import os

def create_test_memory():
    """Create temporary memory for testing"""
    temp_dir = tempfile.mkdtemp()
    test_db_path = os.path.join(temp_dir, "test.db")
    
    return MilvusLiteMemory(
        db_path=test_db_path,
        collection_name="test_collection"
    )

# Use in tests
def test_rag_functionality():
    memory = create_test_memory()
    # Add test documents and run tests
    memory.add_document("Test document", {"category": "test"})
    results = memory.search("test", limit=1)
    assert len(results["documents"]) == 1
```

### Debug Configuration
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Create memory with debug info
memory = MilvusLiteMemory(
    db_path="./debug.db",
    collection_name="debug_collection",
    index_type="HNSW"  # Good for debugging
)

# Monitor database growth
print(f"Database size: {memory.get_database_size()} bytes")
stats = memory.get_collection_stats()
print(f"Document count: {stats['row_count']}")
```

## Best Practices

1. **Database Location**: Store databases in a dedicated data directory
2. **Backup Strategy**: Implement regular backups for important data
3. **Resource Management**: Monitor database size and system resources
4. **Error Handling**: Handle file I/O errors and database corruption
5. **Testing**: Use temporary databases for unit tests
6. **Version Control**: Don't commit database files to version control
7. **Documentation**: Document schema and metadata conventions
8. **Migration Planning**: Plan for eventual migration to full Milvus if needed

## Troubleshooting

### Common Issues

1. **Database File Errors**
   - Check file permissions and disk space
   - Ensure directory exists before creating database
   - Handle concurrent access properly

2. **Performance Issues**
   - Monitor database size relative to available memory
   - Consider index type optimization for dataset size
   - Batch operations for better throughput

3. **Memory Usage**
   - Use appropriate index parameters for available RAM
   - Monitor system memory usage
   - Consider data compression techniques

4. **Data Corruption**
   - Implement proper backup and recovery procedures
   - Handle application crashes gracefully
   - Use database validation tools

This comprehensive guide provides everything needed to integrate Milvus Lite with Swarms agents for local, lightweight RAG applications using the unified LiteLLM embeddings approach.