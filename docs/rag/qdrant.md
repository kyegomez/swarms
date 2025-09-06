# Qdrant RAG Integration

This example demonstrates how to integrate Qdrant vector database with Swarms agents for Retrieval-Augmented Generation (RAG). Qdrant is a high-performance vector database that enables agents to store, index, and retrieve documents using semantic similarity search for enhanced context and more accurate responses.

## Prerequisites

- Python 3.7+
- OpenAI API key
- Swarms library
- Qdrant client and swarms-memory

## Installation

```bash
pip install qdrant-client fastembed swarms-memory litellm
```

> **Note**: The `litellm` package is required for using LiteLLM provider models like OpenAI, Azure, Cohere, etc.

## Tutorial Steps

### Step 1: Install Swarms

First, install the latest version of Swarms:

```bash
pip3 install -U swarms
```

### Step 2: Environment Setup

Set up your environment variables in a `.env` file:

```plaintext
OPENAI_API_KEY="your-api-key-here"
QDRANT_URL="https://your-cluster.qdrant.io"
QDRANT_API_KEY="your-api-key"
WORKSPACE_DIR="agent_workspace"
```

### Step 3: Choose Deployment

Select your Qdrant deployment option:

- **In-memory**: For testing and development (data is not persisted)
- **Local server**: For production deployments with persistent storage
- **Qdrant Cloud**: Managed cloud service (recommended for production)

### Step 4: Configure Database

Set up the vector database wrapper with your preferred embedding model and collection settings

### Step 5: Add Documents

Load documents using individual or batch processing methods

### Step 6: Create Agent

Initialize your agent with RAG capabilities and start querying

## Code

### Basic Setup with Individual Document Processing

```python
from qdrant_client import QdrantClient, models
from swarms import Agent
from swarms_memory import QdrantDB
import os

# Client Configuration Options

# Option 1: In-memory (testing only - data is NOT persisted)
# ":memory:" creates a temporary in-memory database that's lost when program ends
client = QdrantClient(":memory:")

# Option 2: Local Qdrant Server
# Requires: docker run -p 6333:6333 qdrant/qdrant
# client = QdrantClient(host="localhost", port=6333)

# Option 3: Qdrant Cloud (recommended for production)
# Get credentials from https://cloud.qdrant.io
# client = QdrantClient(
#     url=os.getenv("QDRANT_URL"),  # e.g., "https://xyz-abc.eu-central.aws.cloud.qdrant.io"
#     api_key=os.getenv("QDRANT_API_KEY")  # Your Qdrant Cloud API key
# )

# Create vector database wrapper
rag_db = QdrantDB(
    client=client,
    embedding_model="text-embedding-3-small",
    collection_name="knowledge_base",
    distance=models.Distance.COSINE,
    n_results=3
)

# Add documents to the knowledge base
documents = [
    "Qdrant is a vector database optimized for similarity search and AI applications.",
    "RAG combines retrieval and generation for more accurate AI responses.",
    "Vector embeddings enable semantic search across documents.",
    "The swarms framework supports multiple memory backends including Qdrant."
]

# Method 1: Add documents individually
for doc in documents:
    rag_db.add(doc)

# Create agent with RAG capabilities
agent = Agent(
    agent_name="RAG-Agent",
    agent_description="Agent with Qdrant-powered RAG for enhanced knowledge retrieval",
    model_name="gpt-4o",
    max_loops=1,
    dynamic_temperature_enabled=True,
    long_term_memory=rag_db
)

# Query with RAG
try:
    response = agent.run("What is Qdrant and how does it relate to RAG?")
    print(response)
except Exception as e:
    print(f"Error during query: {e}")
    # Handle error appropriately
```

### Advanced Setup with Batch Processing and Metadata

```python
from qdrant_client import QdrantClient, models
from swarms import Agent
from swarms_memory import QdrantDB
import os

# Initialize client (using in-memory for this example)
client = QdrantClient(":memory:")

# Create vector database wrapper
rag_db = QdrantDB(
    client=client,
    embedding_model="text-embedding-3-small",
    collection_name="advanced_knowledge_base",
    distance=models.Distance.COSINE,
    n_results=3
)

# Method 2: Batch add documents (more efficient for large datasets)
# Example with metadata
documents_with_metadata = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing enables computers to understand human language.",
    "Computer vision allows machines to interpret visual information.",
    "Reinforcement learning learns through interaction with an environment."
]

metadata = [
    {"category": "AI", "difficulty": "beginner", "topic": "overview"},
    {"category": "ML", "difficulty": "intermediate", "topic": "neural_networks"},
    {"category": "NLP", "difficulty": "intermediate", "topic": "language"},
    {"category": "CV", "difficulty": "advanced", "topic": "vision"},
    {"category": "RL", "difficulty": "advanced", "topic": "learning"}
]

# Batch add with metadata
doc_ids = rag_db.batch_add(documents_with_metadata, metadata=metadata, batch_size=3)
print(f"Added {len(doc_ids)} documents in batch")

# Query with metadata return
results_with_metadata = rag_db.query(
    "What is artificial intelligence?", 
    n_results=3, 
    return_metadata=True
)

for i, result in enumerate(results_with_metadata):
    print(f"\nResult {i+1}:")
    print(f"  Document: {result['document']}")
    print(f"  Category: {result['category']}")
    print(f"  Difficulty: {result['difficulty']}")
    print(f"  Topic: {result['topic']}")
    print(f"  Score: {result['score']:.4f}")

# Create agent with RAG capabilities
agent = Agent(
    agent_name="Advanced-RAG-Agent",
    agent_description="Advanced agent with metadata-enhanced RAG capabilities",
    model_name="gpt-4o",
    max_loops=1,
    dynamic_temperature_enabled=True,
    long_term_memory=rag_db
)

# Query with enhanced context
response = agent.run("Explain the relationship between machine learning and artificial intelligence")
print(response)
```

## Production Setup

### Setting up Qdrant Cloud

1. Sign up at [cloud.qdrant.io](https://cloud.qdrant.io)
2. Create a cluster
3. Get your cluster URL and API key
4. Set environment variables:

   ```bash
   export QDRANT_URL="https://your-cluster.eu-central.aws.cloud.qdrant.io"
   export QDRANT_API_KEY="your-api-key-here"
   ```

### Running Local Qdrant Server

```bash
# Docker
docker run -p 6333:6333 qdrant/qdrant

# Docker Compose
version: '3.7'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_storage:/qdrant/storage
```

### Production Configuration Example

```python
from qdrant_client import QdrantClient, models
from swarms_memory import QdrantDB
import os
import logging

# Setup logging for production monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Connect to Qdrant server with proper error handling
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        api_key=os.getenv("QDRANT_API_KEY"),  # Use environment variable
        timeout=30  # 30 second timeout
    )
    
    # Production RAG configuration with enhanced settings
    rag_db = QdrantDB(
        client=client,
        embedding_model="text-embedding-3-large",  # Higher quality embeddings
        collection_name="production_knowledge",
        distance=models.Distance.COSINE,
        n_results=10,
        api_key=os.getenv("OPENAI_API_KEY")  # Secure API key handling
    )
    
    logger.info("Successfully initialized production RAG database")
    
except Exception as e:
    logger.error(f"Failed to initialize RAG database: {e}")
    raise
```

## Configuration Options

### Distance Metrics

| Metric | Description | Best For |
|--------|-------------|----------|
| **COSINE** | Cosine similarity (default) | Normalized embeddings, text similarity |
| **EUCLIDEAN** | Euclidean distance | Absolute distance measurements |
| **DOT** | Dot product | Maximum inner product search |

### Embedding Model Options

#### LiteLLM Provider Models (Recommended)

| Model | Provider | Dimensions | Description |
|-------|----------|------------|-------------|
| `text-embedding-3-small` | OpenAI | 1536 | Efficient, cost-effective |
| `text-embedding-3-large` | OpenAI | 3072 | Best quality |
| `azure/your-deployment` | Azure | Variable | Azure OpenAI embeddings |
| `cohere/embed-english-v3.0` | Cohere | 1024 | Advanced language understanding |
| `voyage/voyage-3-large` | Voyage AI | 1024 | High-quality embeddings |

#### SentenceTransformer Models

| Model | Dimensions | Description |
|-------|------------|-------------|
| `all-MiniLM-L6-v2` | 384 | Fast, general-purpose |
| `all-mpnet-base-v2` | 768 | Higher quality |
| `all-roberta-large-v1` | 1024 | Best quality |

#### Usage Example

```python
# OpenAI embeddings (default example)
rag_db = QdrantDB(
    client=client,
    embedding_model="text-embedding-3-small",
    collection_name="openai_collection"
)
```

> **Note**: QdrantDB supports all LiteLLM provider models (Azure, Cohere, Voyage AI, etc.), SentenceTransformer models, and custom embedding functions. See the embedding model options table above for the complete list.

## Use Cases

### Document Q&A System

Create an intelligent document question-answering system:

```python
# Load company documents into Qdrant
company_documents = [
    "Company policy on remote work allows flexible scheduling with core hours 10 AM - 3 PM.",
    "API documentation: Use POST /api/v1/users to create new user accounts.",
    "Product specifications: Our software supports Windows, Mac, and Linux platforms."
]

for doc in company_documents:
    rag_db.add(doc)

# Agent can now answer questions using the documents
agent = Agent(
    agent_name="Company-DocQA-Agent",
    agent_description="Intelligent document Q&A system for company information",
    model_name="gpt-4o",
    long_term_memory=rag_db
)

answer = agent.run("What is the company policy on remote work?")
print(answer)
```

### Knowledge Base Management

Build a comprehensive knowledge management system:

```python
class KnowledgeBaseAgent:
    def __init__(self):
        self.client = QdrantClient(":memory:")
        self.rag_db = QdrantDB(
            client=self.client,
            embedding_model="text-embedding-3-small",
            collection_name="knowledge_base",
            n_results=5
        )
        self.agent = Agent(
            agent_name="KB-Management-Agent",
            agent_description="Knowledge base management and retrieval system",
            model_name="gpt-4o",
            long_term_memory=self.rag_db
        )
    
    def add_knowledge(self, text: str, metadata: dict = None):
        """Add new knowledge to the base"""
        if metadata:
            return self.rag_db.batch_add([text], metadata=[metadata])
        return self.rag_db.add(text)
    
    def query(self, question: str):
        """Query the knowledge base"""
        return self.agent.run(question)
    
    def bulk_import(self, documents: list, metadata_list: list = None):
        """Import multiple documents efficiently"""
        return self.rag_db.batch_add(documents, metadata=metadata_list, batch_size=50)

# Usage
kb = KnowledgeBaseAgent()
kb.add_knowledge("Python is a high-level programming language.", {"category": "programming"})
kb.add_knowledge("Qdrant is optimized for vector similarity search.", {"category": "databases"})
result = kb.query("What programming languages are mentioned?")
print(result)
```

## Best Practices

### Document Processing Strategy

| Practice | Recommendation | Details |
|----------|----------------|---------|
| **Chunking** | 200-500 tokens | Split large documents into optimal chunks for retrieval |
| **Overlap** | 20-50 tokens | Maintain context between consecutive chunks |
| **Preprocessing** | Clean & normalize | Remove noise and standardize text format |

### Collection Organization

| Practice | Recommendation | Details |
|----------|----------------|---------|
| **Separation** | Type-based collections | Use separate collections for docs, policies, code, etc. |
| **Naming** | Consistent conventions | Follow clear, descriptive naming patterns |
| **Lifecycle** | Update strategies | Plan for document versioning and updates |

### Embedding Model Selection

| Environment | Recommended Model | Use Case |
|-------------|-------------------|----------|
| **Development** | `all-MiniLM-L6-v2` | Fast iteration and testing |
| **Production** | `text-embedding-3-small/large` | High-quality production deployment |
| **Specialized** | Domain-specific models | Industry or domain-focused applications |

### Performance Optimization

| Setting | Recommendation | Rationale |
|---------|----------------|-----------|
| **Retrieval Count** | Start with 3-5 results | Balance relevance with performance |
| **Batch Operations** | Use `batch_add()` | Efficient bulk document processing |
| **Metadata** | Strategic storage | Enable filtering and enhanced context |

### Production Deployment

| Component | Best Practice | Implementation |
|-----------|---------------|----------------|
| **Storage** | Persistent server | Use Qdrant Cloud or self-hosted server |
| **Error Handling** | Robust mechanisms | Implement retry logic and graceful failures |
| **Monitoring** | Performance tracking | Monitor metrics and embedding quality |

## Performance Tips

- **Development**: Use in-memory mode for rapid prototyping and testing
- **Production**: Deploy dedicated Qdrant server with appropriate resource allocation
- **Scalability**: Use batch operations for adding multiple documents efficiently
- **Memory Management**: Monitor memory usage with large document collections
- **API Usage**: Consider rate limits when using cloud-based embedding services
- **Caching**: Implement caching strategies for frequently accessed documents

## Customization

You can modify the system configuration to create specialized RAG agents for different use cases:

| Use Case | Configuration | Description |
|----------|---------------|-------------|
| **Technical Documentation** | High n_results (10-15), precise embeddings | Comprehensive technical Q&A |
| **Customer Support** | Fast embeddings, metadata filtering | Quick response with categorization |
| **Research Assistant** | Large embedding model, broad retrieval | Deep analysis and synthesis |
| **Code Documentation** | Code-specific embeddings, semantic chunking | Programming-focused assistance |

## Related Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Swarms Memory GitHub Repository](https://github.com/The-Swarm-Corporation/swarms-memory)
- [Agent Documentation](../agents/new_agent.md)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Vector Database Concepts](https://qdrant.tech/documentation/concepts/)