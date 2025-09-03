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
Weaviate Local RAG Integration with Swarms Agent

This example demonstrates how to integrate self-hosted Weaviate as a customizable
vector database for RAG operations with full local control.
"""

import weaviate
from typing import List, Dict, Any, Optional
from swarms import Agent
from litellm import embedding
import uuid
from datetime import datetime

class WeaviateLocalMemory:
    """Weaviate Local-based memory system for RAG operations"""
    
    def __init__(self, 
                 url: str = "http://localhost:8080",
                 class_name: str = "LocalDocument",
                 embedding_model: str = "text-embedding-3-small",
                 use_builtin_vectorization: bool = False,
                 auth_config: Optional[Dict] = None):
        """
        Initialize Weaviate Local memory system
        
        Args:
            url: Weaviate server URL
            class_name: Name of the Weaviate class
            embedding_model: LiteLLM embedding model name
            use_builtin_vectorization: Use Weaviate's built-in vectorization
            auth_config: Authentication configuration
        """
        self.url = url
        self.class_name = class_name
        self.embedding_model = embedding_model
        self.use_builtin_vectorization = use_builtin_vectorization
        
        # Initialize client
        self.client = self._create_client(auth_config)
        
        # Create schema
        self._create_schema()
        
    def _create_client(self, auth_config: Optional[Dict] = None):
        """Create Weaviate local client"""
        client_config = {"url": self.url}
        
        if auth_config:
            if auth_config.get("type") == "api_key":
                client_config["auth_client_secret"] = weaviate.AuthApiKey(
                    api_key=auth_config["api_key"]
                )
            elif auth_config.get("type") == "username_password":
                client_config["auth_client_secret"] = weaviate.AuthClientPassword(
                    username=auth_config["username"],
                    password=auth_config["password"]
                )
        
        # Add API keys for modules
        additional_headers = {}
        if "OPENAI_API_KEY" in os.environ:
            additional_headers["X-OpenAI-Api-Key"] = os.environ["OPENAI_API_KEY"]
        
        if additional_headers:
            client_config["additional_headers"] = additional_headers
        
        client = weaviate.Client(**client_config)
        
        # Test connection
        try:
            client.schema.get()
            print(f"Connected to Weaviate Local: {self.url}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Weaviate: {e}")
        
        return client
    
    def _create_schema(self):
        """Create Weaviate class schema"""
        schema = self.client.schema.get()
        existing_classes = [c["class"] for c in schema.get("classes", [])]
        
        if self.class_name in existing_classes:
            print(f"Class '{self.class_name}' already exists")
            return
        
        # Define comprehensive schema
        class_obj = {
            "class": self.class_name,
            "description": "Local document class for Swarms RAG operations",
            "vectorizer": "none" if not self.use_builtin_vectorization else "text2vec-openai",
            "properties": [
                {
                    "name": "text",
                    "dataType": ["text"],
                    "description": "Document content",
                    "indexFilterable": True,
                    "indexSearchable": True,
                    "tokenization": "word"
                },
                {
                    "name": "title",
                    "dataType": ["string"],
                    "description": "Document title",
                    "indexFilterable": True
                },
                {
                    "name": "category",
                    "dataType": ["string"],
                    "description": "Document category",
                    "indexFilterable": True
                },
                {
                    "name": "tags",
                    "dataType": ["string[]"],
                    "description": "Document tags",
                    "indexFilterable": True
                },
                {
                    "name": "author",
                    "dataType": ["string"],
                    "description": "Document author",
                    "indexFilterable": True
                },
                {
                    "name": "created_at",
                    "dataType": ["date"],
                    "description": "Creation date"
                },
                {
                    "name": "metadata",
                    "dataType": ["object"],
                    "description": "Additional metadata"
                }
            ]
        }
        
        self.client.schema.create_class(class_obj)
        print(f"Created local class '{self.class_name}'")
    
    def add_documents(self, documents: List[Dict]) -> List[str]:
        """Add documents with rich metadata to Weaviate Local"""
        doc_ids = []
        
        with self.client.batch as batch:
            batch.batch_size = 100
            
            for doc_data in documents:
                doc_id = str(uuid.uuid4())
                doc_ids.append(doc_id)
                
                # Prepare properties
                properties = {
                    "text": doc_data.get("text", ""),
                    "title": doc_data.get("title", ""),
                    "category": doc_data.get("category", ""),
                    "tags": doc_data.get("tags", []),
                    "author": doc_data.get("author", ""),
                    "created_at": doc_data.get("created_at", datetime.now().isoformat()),
                    "metadata": doc_data.get("metadata", {})
                }
                
                batch_obj = {
                    "class": self.class_name,
                    "id": doc_id,
                    "properties": properties
                }
                
                # Add vector if using external embeddings
                if not self.use_builtin_vectorization:
                    text_content = doc_data.get("text", "")
                    if text_content:
                        embedding_vec = self._get_embeddings([text_content])[0]
                        batch_obj["vector"] = embedding_vec
                
                batch.add_data_object(**batch_obj)
        
        print(f"Added {len(documents)} documents to local Weaviate")
        return doc_ids
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using LiteLLM"""
        response = embedding(model=self.embedding_model, input=texts)
        return [item["embedding"] for item in response["data"]]
    
    def search(self, query: str, limit: int = 3, **kwargs) -> Dict[str, Any]:
        """Search documents with flexible filtering"""
        # Build query
        query_builder = (
            self.client.query
            .get(self.class_name, ["text", "title", "category", "tags", "author", "metadata"])
        )
        
        # Add vector search
        if self.use_builtin_vectorization:
            query_builder = query_builder.with_near_text({"concepts": [query]})
        else:
            query_embedding = self._get_embeddings([query])[0]
            query_builder = query_builder.with_near_vector({"vector": query_embedding})
        
        # Add optional filters
        if "where_filter" in kwargs:
            query_builder = query_builder.with_where(kwargs["where_filter"])
        
        # Execute query
        result = (
            query_builder
            .with_limit(limit)
            .with_additional(["certainty", "distance", "id"])
            .do()
        )
        
        # Format results
        formatted_results = {"documents": [], "metadata": [], "scores": [], "ids": []}
        
        if "data" in result and "Get" in result["data"]:
            for item in result["data"]["Get"].get(self.class_name, []):
                formatted_results["documents"].append(item.get("text", ""))
                
                # Combine all metadata
                metadata = {
                    "title": item.get("title", ""),
                    "category": item.get("category", ""),
                    "tags": item.get("tags", []),
                    "author": item.get("author", ""),
                    **item.get("metadata", {})
                }
                formatted_results["metadata"].append(metadata)
                formatted_results["ids"].append(item["_additional"]["id"])
                
                score = item["_additional"].get("certainty", 0.0)
                formatted_results["scores"].append(float(score))
        
        return formatted_results

# Sample usage
memory = WeaviateLocalMemory(
    url="http://localhost:8080",
    class_name="SwarmsLocalKB",
    embedding_model="text-embedding-3-small"
)

# Add rich documents
documents = [
    {
        "text": "Weaviate Local provides full control over vector database deployment and data sovereignty.",
        "title": "Local Deployment Benefits",
        "category": "deployment",
        "tags": ["weaviate", "local", "control"],
        "author": "System",
        "metadata": {"difficulty": "intermediate", "topic": "infrastructure"}
    },
    {
        "text": "Self-hosted Weaviate enables custom configurations and air-gapped deployments for sensitive data.",
        "title": "Security and Compliance",
        "category": "security", 
        "tags": ["security", "compliance", "air-gap"],
        "author": "Admin",
        "metadata": {"difficulty": "advanced", "topic": "security"}
    }
]

# Create agent and add documents
memory.add_documents(documents)

agent = Agent(
    agent_name="Local-Weaviate-Agent",
    agent_description="Agent with self-hosted Weaviate for private RAG operations",
    model_name="gpt-4o",
    max_loops=1,
    dynamic_temperature_enabled=True,
)

def query_local_rag(query: str, limit: int = 3) -> str:
    """Query local Weaviate with RAG"""
    results = memory.search(query, limit=limit)
    
    if not results["documents"]:
        return agent.run(query)
    
    context = "\n".join(results["documents"])
    
    enhanced_prompt = f"""
Based on this local knowledge base context:

{context}

Question: {query}

Provide a comprehensive answer using the context.
"""
    
    return agent.run(enhanced_prompt)

# Example usage
response = query_local_rag("What are the benefits of local Weaviate deployment?")
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