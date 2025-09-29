# Weaviate Local RAG Integration with Swarms

## Overview

Weaviate Local is a self-hosted version of the Weaviate vector database that runs on your own infrastructure. It provides the same powerful GraphQL API, multi-modal capabilities, and AI integrations as Weaviate Cloud, but with full control over data, deployment, and customization. Weaviate Local is ideal for organizations requiring data sovereignty, custom configurations, or air-gapped deployments while maintaining enterprise-grade vector search capabilities.

## Key Features

- **Self-Hosted Control**: Full ownership of data and infrastructure
- **GraphQL API**: Flexible query language for complex data operations
- **Multi-Modal Support**: Built-in support for text, images, and other data types
- **Custom Modules**: Extensible architecture with custom vectorization modules
- **Docker Deployment**: Easy containerized deployment and scaling
- **Schema Flexibility**: Dynamic schema with automatic type inference
- **Hybrid Search**: Combine vector similarity with keyword search
- **Real-time Updates**: Live data updates without service interruption

## Architecture

Weaviate Local integrates with Swarms agents as a self-hosted, customizable vector database:

```
[Agent] -> [Weaviate Local Memory] -> [Local GraphQL + Vector Engine] -> [Custom Results] -> [Retrieved Context]
```

The system provides full control over the deployment environment while maintaining Weaviate's advanced search capabilities.

## Setup & Configuration

### Installation

```bash
# Docker installation (recommended)
docker pull semitechnologies/weaviate:latest

# Python client
pip install weaviate-client
pip install swarms
pip install litellm
```

### Docker Deployment

```yaml
# docker-compose.yml
version: '3.4'
services:
  weaviate:
    command:
    - --host
    - 0.0.0.0
    - --port
    - '8080'
    - --scheme
    - http
    image: semitechnologies/weaviate:1.22.4
    ports:
    - "8080:8080"
    restart: on-failure:0
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      ENABLE_MODULES: 'text2vec-openai,text2vec-cohere,text2vec-huggingface'
      CLUSTER_HOSTNAME: 'node1'
    volumes:
    - weaviate_data:/var/lib/weaviate
volumes:
  weaviate_data:
```

### Environment Variables

```bash
# Local Weaviate connection
export WEAVIATE_URL="http://localhost:8080"

# Optional: Authentication (if enabled)
export WEAVIATE_USERNAME="admin"
export WEAVIATE_PASSWORD="password"

# API keys for built-in modules
export OPENAI_API_KEY="your-openai-key"
export COHERE_API_KEY="your-cohere-key"
export HUGGINGFACE_API_KEY="your-hf-key"
```

## Code Example

```python
"""
Agent with Weaviate Local RAG

This example demonstrates using local Weaviate as a vector database for RAG operations,
allowing agents to store and retrieve documents for enhanced context.
"""

from swarms import Agent
from swarms_memory import WeaviateDB


# Create WeaviateDB wrapper for RAG operations
rag_db = WeaviateDB(
    embedding_model="text-embedding-3-small",
    collection_name="swarms_knowledge",
    cluster_url="http://localhost:8080",  # Local Weaviate instance
    distance_metric="cosine",
)

# Add documents to the knowledge base
documents = [
    "Weaviate is an open-source vector database optimized for similarity search and AI applications.",
    "RAG combines retrieval and generation for more accurate AI responses.",
    "Vector embeddings enable semantic search across documents.",
    "The Swarms framework supports multiple memory backends including Weaviate.",
    "Swarms is the first and most reliable multi-agent production-grade framework.",
    "Kye Gomez is Founder and CEO of Swarms Corporation."
]

# Add documents individually
for doc in documents:
    rag_db.add(doc)

# Create agent with RAG capabilities
agent = Agent(
    agent_name="Weaviate-RAG-Agent",
    agent_description="Swarms Agent with Weaviate-powered RAG for enhanced knowledge retrieval",
    model_name="gpt-4o",
    max_loops=1,
    dynamic_temperature_enabled=True,
    long_term_memory=rag_db
)

# Query with RAG
response = agent.run("What is Weaviate and how does it relate to RAG? Who is the founder of Swarms?")
print(response)
```

## Use Cases

### 1. **Data Sovereignty & Compliance**
- Government and healthcare organizations
- GDPR/HIPAA compliance requirements
- Sensitive data processing

### 2. **Air-Gapped Environments**
- Military and defense applications
- High-security research facilities
- Offline AI systems

### 3. **Custom Infrastructure**
- Specific hardware requirements
- Custom networking configurations
- Specialized security measures

### 4. **Development & Testing**
- Local development environments
- CI/CD integration
- Performance testing

## Deployment Options

### Docker Compose
```yaml
version: '3.4'
services:
  weaviate:
    image: semitechnologies/weaviate:1.22.4
    restart: on-failure:0
    ports:
      - "8080:8080"
    environment:
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      ENABLE_MODULES: 'text2vec-openai,backup-filesystem'
    volumes:
      - ./weaviate_data:/var/lib/weaviate
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: weaviate
spec:
  replicas: 1
  selector:
    matchLabels:
      app: weaviate
  template:
    metadata:
      labels:
        app: weaviate
    spec:
      containers:
      - name: weaviate
        image: semitechnologies/weaviate:1.22.4
        ports:
        - containerPort: 8080
        env:
        - name: PERSISTENCE_DATA_PATH
          value: '/var/lib/weaviate'
        volumeMounts:
        - name: weaviate-storage
          mountPath: /var/lib/weaviate
      volumes:
      - name: weaviate-storage
        persistentVolumeClaim:
          claimName: weaviate-pvc
```

## Best Practices

1. **Resource Planning**: Allocate sufficient memory and storage for your dataset
2. **Backup Strategy**: Implement regular backups using Weaviate's backup modules
3. **Monitoring**: Set up health checks and performance monitoring
4. **Security**: Configure authentication and network security appropriately
5. **Scaling**: Plan for horizontal scaling with clustering if needed
6. **Updates**: Establish update procedures for Weaviate versions
7. **Data Migration**: Plan migration strategies for schema changes

This guide covers the essentials of deploying and integrating Weaviate Local with Swarms agents for private, self-controlled RAG applications.