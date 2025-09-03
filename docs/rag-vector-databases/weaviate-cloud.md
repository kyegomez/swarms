# Weaviate Cloud RAG Integration with Swarms

## Overview

Weaviate Cloud is a fully managed vector database service offering enterprise-grade vector search capabilities with built-in AI integrations. It combines GraphQL APIs with vector search, automatic schema inference, and native ML model integrations. Weaviate Cloud excels in multi-modal search, semantic understanding, and complex relationship modeling, making it ideal for sophisticated RAG applications requiring both vector similarity and graph-like data relationships.

## Key Features

- **GraphQL API**: Flexible query language for complex data retrieval
- **Multi-modal Search**: Support for text, images, and other data types
- **Built-in Vectorization**: Automatic embedding generation with various models
- **Schema Flexibility**: Dynamic schema with automatic type inference
- **Hybrid Search**: Combine vector similarity with keyword search
- **Graph Relationships**: Model complex data relationships
- **Enterprise Security**: SOC 2 compliance with role-based access control
- **Global Distribution**: Multi-region deployment with low latency

## Architecture

Weaviate Cloud integrates with Swarms agents as an intelligent, multi-modal vector database:

```
[Agent] -> [Weaviate Cloud Memory] -> [GraphQL + Vector Search] -> [Multi-modal Results] -> [Retrieved Context]
```

The system leverages Weaviate's GraphQL interface and built-in AI capabilities for sophisticated semantic search and relationship queries.

## Setup & Configuration

### Installation

```bash
pip install weaviate-client
pip install swarms
pip install litellm
```

### Environment Variables

```bash
# Weaviate Cloud credentials
export WEAVIATE_URL="https://your-cluster.weaviate.network"
export WEAVIATE_API_KEY="your-api-key"

# Optional: OpenAI API key (for built-in vectorization)
export OPENAI_API_KEY="your-openai-api-key"

# Optional: Additional model API keys
export COHERE_API_KEY="your-cohere-key"
export HUGGINGFACE_API_KEY="your-hf-key"
```

### Dependencies

- `weaviate-client>=4.4.0`
- `swarms`
- `litellm`
- `numpy`

## Code Example

```python
"""
Weaviate Cloud RAG Integration with Swarms Agent

This example demonstrates how to integrate Weaviate Cloud as a multi-modal vector database
for RAG operations with Swarms agents using both built-in and LiteLLM embeddings.
"""

import os
import weaviate
from typing import List, Dict, Any, Optional
from swarms import Agent
from litellm import embedding
import uuid
from datetime import datetime

class WeaviateCloudMemory:
    """Weaviate Cloud-based memory system for RAG operations"""
    
    def __init__(self, 
                 class_name: str = "SwarmsDocument",
                 embedding_model: str = "text-embedding-3-small",
                 use_builtin_vectorization: bool = False,
                 vectorizer: str = "text2vec-openai"):
        """
        Initialize Weaviate Cloud memory system
        
        Args:
            class_name: Name of the Weaviate class (collection)
            embedding_model: LiteLLM embedding model name
            use_builtin_vectorization: Use Weaviate's built-in vectorization
            vectorizer: Built-in vectorizer module (if used)
        """
        self.class_name = class_name
        self.embedding_model = embedding_model
        self.use_builtin_vectorization = use_builtin_vectorization
        self.vectorizer = vectorizer
        
        # Initialize Weaviate client
        self.client = self._create_client()
        
        # Create class schema if it doesn't exist
        self._create_schema()
        
    def _create_client(self):
        """Create Weaviate Cloud client"""
        url = os.getenv("WEAVIATE_URL")
        api_key = os.getenv("WEAVIATE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if not url:
            raise ValueError("WEAVIATE_URL must be set")
        
        auth_config = None
        if api_key:
            auth_config = weaviate.AuthApiKey(api_key=api_key)
        
        # Additional headers for API keys
        additional_headers = {}
        if openai_key:
            additional_headers["X-OpenAI-Api-Key"] = openai_key
        
        client = weaviate.Client(
            url=url,
            auth_client_secret=auth_config,
            additional_headers=additional_headers
        )
        
        print(f"Connected to Weaviate Cloud: {url}")
        return client
        
    def _create_schema(self):
        """Create Weaviate class schema"""
        # Check if class already exists
        schema = self.client.schema.get()
        existing_classes = [c["class"] for c in schema.get("classes", [])]
        
        if self.class_name in existing_classes:
            print(f"Class '{self.class_name}' already exists")
            return
        
        # Define class schema
        class_obj = {
            "class": self.class_name,
            "description": "Document class for Swarms RAG operations",
            "properties": [
                {
                    "name": "text",
                    "dataType": ["text"],
                    "description": "The document content",
                    "indexFilterable": True,
                    "indexSearchable": True
                },
                {
                    "name": "category",
                    "dataType": ["string"],
                    "description": "Document category",
                    "indexFilterable": True
                },
                {
                    "name": "topic",
                    "dataType": ["string"],
                    "description": "Document topic",
                    "indexFilterable": True
                },
                {
                    "name": "difficulty",
                    "dataType": ["string"],
                    "description": "Content difficulty level",
                    "indexFilterable": True
                },
                {
                    "name": "metadata",
                    "dataType": ["object"],
                    "description": "Additional metadata"
                },
                {
                    "name": "timestamp",
                    "dataType": ["date"],
                    "description": "Creation timestamp"
                }
            ]
        }
        
        # Add vectorizer configuration if using built-in
        if self.use_builtin_vectorization:
            class_obj["vectorizer"] = self.vectorizer
            if self.vectorizer == "text2vec-openai":
                class_obj["moduleConfig"] = {
                    "text2vec-openai": {
                        "model": "ada",
                        "modelVersion": "002",
                        "type": "text"
                    }
                }
        else:
            class_obj["vectorizer"] = "none"
        
        # Create the class
        self.client.schema.create_class(class_obj)
        print(f"Created class '{self.class_name}'")
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using LiteLLM (when not using built-in vectorization)"""
        if self.use_builtin_vectorization:
            return None  # Weaviate will handle vectorization
        
        response = embedding(
            model=self.embedding_model,
            input=texts
        )
        return [item["embedding"] for item in response["data"]]
    
    def add_documents(self, 
                     documents: List[str], 
                     metadata: List[Dict] = None,
                     batch_size: int = 100) -> List[str]:
        """Add multiple documents to Weaviate"""
        if metadata is None:
            metadata = [{}] * len(documents)
        
        # Generate embeddings if not using built-in vectorization
        embeddings = self._get_embeddings(documents) if not self.use_builtin_vectorization else None
        
        # Prepare objects for batch import
        objects = []
        doc_ids = []
        
        for i, (doc, meta) in enumerate(zip(documents, metadata)):
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)
            
            obj = {
                "class": self.class_name,
                "id": doc_id,
                "properties": {
                    "text": doc,
                    "category": meta.get("category", ""),
                    "topic": meta.get("topic", ""),
                    "difficulty": meta.get("difficulty", ""),
                    "metadata": meta,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Add vector if using external embeddings
            if embeddings and i < len(embeddings):
                obj["vector"] = embeddings[i]
            
            objects.append(obj)
        
        # Batch import
        with self.client.batch as batch:
            batch.batch_size = batch_size
            for obj in objects:
                batch.add_data_object(**obj)
        
        print(f"Added {len(documents)} documents to Weaviate Cloud")
        return doc_ids
    
    def add_document(self, document: str, metadata: Dict = None) -> str:
        """Add a single document to Weaviate"""
        result = self.add_documents([document], [metadata or {}])
        return result[0] if result else None
    
    def search(self, 
               query: str,
               limit: int = 3,
               where_filter: Dict = None,
               certainty: float = None,
               distance: float = None) -> Dict[str, Any]:
        """Search for similar documents in Weaviate Cloud"""
        
        # Build GraphQL query
        query_builder = (
            self.client.query
            .get(self.class_name, ["text", "category", "topic", "difficulty", "metadata"])
        )
        
        # Add vector search
        if self.use_builtin_vectorization:
            query_builder = query_builder.with_near_text({"concepts": [query]})
        else:
            # Generate query embedding
            query_embedding = self._get_embeddings([query])[0]
            query_builder = query_builder.with_near_vector({"vector": query_embedding})
        
        # Add filters
        if where_filter:
            query_builder = query_builder.with_where(where_filter)
        
        # Add certainty/distance threshold
        if certainty is not None:
            query_builder = query_builder.with_certainty(certainty)
        elif distance is not None:
            query_builder = query_builder.with_distance(distance)
        
        # Set limit
        query_builder = query_builder.with_limit(limit)
        
        # Add additional fields for scoring
        query_builder = query_builder.with_additional(["certainty", "distance", "id"])
        
        # Execute query
        result = query_builder.do()
        
        # Format results
        formatted_results = {
            "documents": [],
            "metadata": [],
            "scores": [],
            "ids": []
        }
        
        if "data" in result and "Get" in result["data"] and self.class_name in result["data"]["Get"]:
            for item in result["data"]["Get"][self.class_name]:
                formatted_results["documents"].append(item.get("text", ""))
                formatted_results["metadata"].append(item.get("metadata", {}))
                formatted_results["ids"].append(item["_additional"]["id"])
                
                # Use certainty (higher is better) or distance (lower is better)
                score = item["_additional"].get("certainty", 
                        1.0 - item["_additional"].get("distance", 1.0))
                formatted_results["scores"].append(float(score))
        
        return formatted_results
    
    def hybrid_search(self, 
                     query: str,
                     limit: int = 3,
                     alpha: float = 0.5,
                     where_filter: Dict = None) -> Dict[str, Any]:
        """Perform hybrid search (vector + keyword) in Weaviate"""
        
        query_builder = (
            self.client.query
            .get(self.class_name, ["text", "category", "topic", "difficulty", "metadata"])
            .with_hybrid(query=query, alpha=alpha)  # alpha: 0=keyword, 1=vector
        )
        
        if where_filter:
            query_builder = query_builder.with_where(where_filter)
        
        query_builder = (
            query_builder
            .with_limit(limit)
            .with_additional(["score", "id"])
        )
        
        result = query_builder.do()
        
        # Format results (similar to search method)
        formatted_results = {
            "documents": [],
            "metadata": [],
            "scores": [],
            "ids": []
        }
        
        if "data" in result and "Get" in result["data"] and self.class_name in result["data"]["Get"]:
            for item in result["data"]["Get"][self.class_name]:
                formatted_results["documents"].append(item.get("text", ""))
                formatted_results["metadata"].append(item.get("metadata", {}))
                formatted_results["ids"].append(item["_additional"]["id"])
                formatted_results["scores"].append(float(item["_additional"].get("score", 0.0)))
        
        return formatted_results
    
    def delete_documents(self, where_filter: Dict) -> bool:
        """Delete documents matching filter"""
        result = self.client.batch.delete_objects(
            class_name=self.class_name,
            where=where_filter
        )
        return "results" in result and "successful" in result["results"]
    
    def get_class_info(self) -> Dict[str, Any]:
        """Get class statistics and information"""
        # Get class schema
        schema = self.client.schema.get(self.class_name)
        
        # Get object count (approximate)
        result = (
            self.client.query
            .aggregate(self.class_name)
            .with_meta_count()
            .do()
        )
        
        count = 0
        if "data" in result and "Aggregate" in result["data"]:
            agg_data = result["data"]["Aggregate"][self.class_name][0]
            count = agg_data.get("meta", {}).get("count", 0)
        
        return {
            "class_name": self.class_name,
            "object_count": count,
            "properties": len(schema.get("properties", [])),
            "vectorizer": schema.get("vectorizer"),
            "description": schema.get("description", "")
        }

# Initialize Weaviate Cloud memory
# Option 1: Using LiteLLM embeddings
memory = WeaviateCloudMemory(
    class_name="SwarmsKnowledgeBase",
    embedding_model="text-embedding-3-small",
    use_builtin_vectorization=False
)

# Option 2: Using Weaviate's built-in vectorization
# memory = WeaviateCloudMemory(
#     class_name="SwarmsKnowledgeBase",
#     use_builtin_vectorization=True,
#     vectorizer="text2vec-openai"
# )

# Sample documents for the knowledge base
documents = [
    "Weaviate Cloud is a fully managed vector database with GraphQL API and built-in AI integrations.",
    "RAG systems benefit from Weaviate's multi-modal search and relationship modeling capabilities.",
    "Vector embeddings in Weaviate can be generated using built-in modules or external models.",
    "The Swarms framework leverages Weaviate's GraphQL interface for flexible data queries.",
    "LiteLLM integration provides unified access to embedding models across different providers.",
    "Hybrid search in Weaviate combines vector similarity with traditional keyword search.",
    "GraphQL enables complex queries with filtering, aggregation, and relationship traversal.",
    "Multi-modal capabilities allow searching across text, images, and other data types simultaneously.",
]

# Rich metadata for advanced filtering and organization
metadatas = [
    {"category": "database", "topic": "weaviate", "difficulty": "intermediate", "type": "overview"},
    {"category": "ai", "topic": "rag", "difficulty": "intermediate", "type": "concept"},
    {"category": "ml", "topic": "embeddings", "difficulty": "beginner", "type": "concept"},
    {"category": "framework", "topic": "swarms", "difficulty": "beginner", "type": "integration"},
    {"category": "library", "topic": "litellm", "difficulty": "beginner", "type": "tool"},
    {"category": "search", "topic": "hybrid", "difficulty": "advanced", "type": "feature"},
    {"category": "api", "topic": "graphql", "difficulty": "advanced", "type": "interface"},
    {"category": "multimodal", "topic": "search", "difficulty": "advanced", "type": "capability"},
]

# Add documents to Weaviate Cloud
print("Adding documents to Weaviate Cloud...")
doc_ids = memory.add_documents(documents, metadatas)
print(f"Successfully added {len(doc_ids)} documents")

# Display class information
info = memory.get_class_info()
print(f"Class info: {info['object_count']} objects, {info['properties']} properties")

# Create Swarms agent with Weaviate Cloud RAG
agent = Agent(
    agent_name="Weaviate-RAG-Agent",
    agent_description="Intelligent agent with Weaviate Cloud multi-modal RAG capabilities",
    model_name="gpt-4o",
    max_loops=1,
    dynamic_temperature_enabled=True,
)

def query_with_weaviate_rag(query_text: str, 
                           limit: int = 3, 
                           search_type: str = "vector",
                           where_filter: Dict = None,
                           alpha: float = 0.5):
    """Query with RAG using Weaviate's advanced search capabilities"""
    print(f"\nQuerying ({search_type}): {query_text}")
    if where_filter:
        print(f"Filter: {where_filter}")
    
    # Choose search method
    if search_type == "hybrid":
        results = memory.hybrid_search(
            query=query_text,
            limit=limit,
            alpha=alpha,
            where_filter=where_filter
        )
    else:  # vector search
        results = memory.search(
            query=query_text,
            limit=limit,
            where_filter=where_filter
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
    # Test vector search
    print("=== Vector Search Queries ===")
    queries = [
        "What is Weaviate Cloud and its key features?",
        "How does RAG work with vector databases?",
        "What are hybrid search capabilities?",
        "How does GraphQL enhance database queries?",
    ]
    
    for query in queries:
        response = query_with_weaviate_rag(query, limit=3, search_type="vector")
        print(f"Answer: {response}\n")
        print("-" * 80)
    
    # Test hybrid search
    print("\n=== Hybrid Search Queries ===")
    hybrid_queries = [
        "advanced search features",
        "GraphQL API capabilities",
        "multi-modal AI applications",
    ]
    
    for query in hybrid_queries:
        response = query_with_weaviate_rag(
            query, 
            limit=3, 
            search_type="hybrid",
            alpha=0.7  # More weight on vector search
        )
        print(f"Hybrid Answer: {response}\n")
        print("-" * 80)
    
    # Test filtered search
    print("\n=== Filtered Search Queries ===")
    
    # Filter for advanced topics
    advanced_filter = {
        "path": ["difficulty"],
        "operator": "Equal",
        "valueText": "advanced"
    }
    response = query_with_weaviate_rag(
        "What are advanced capabilities?",
        limit=3,
        where_filter=advanced_filter
    )
    print(f"Advanced topics: {response}\n")
    
    # Filter for database and API categories
    category_filter = {
        "operator": "Or",
        "operands": [
            {
                "path": ["category"],
                "operator": "Equal", 
                "valueText": "database"
            },
            {
                "path": ["category"],
                "operator": "Equal",
                "valueText": "api"
            }
        ]
    }
    response = query_with_weaviate_rag(
        "Tell me about database and API features",
        limit=3,
        where_filter=category_filter
    )
    print(f"Database & API features: {response}\n")
    
    # Add new document dynamically
    print("=== Dynamic Document Addition ===")
    new_doc = "Weaviate's schema flexibility allows automatic type inference and dynamic property addition."
    new_meta = {
        "category": "schema", 
        "topic": "flexibility", 
        "difficulty": "intermediate",
        "type": "feature"
    }
    memory.add_document(new_doc, new_meta)
    
    # Query about schema flexibility
    response = query_with_weaviate_rag("How does schema flexibility work?")
    print(f"Schema flexibility: {response}\n")
    
    # Display final class information
    final_info = memory.get_class_info()
    print(f"Final class info: {final_info}")
```

## Use Cases

### 1. Multi-Modal Knowledge Systems
- **Scenario**: Applications requiring search across text, images, and other media
- **Benefits**: Native multi-modal support, unified search interface
- **Best For**: Content management, media libraries, educational platforms

### 2. Complex Relationship Modeling
- **Scenario**: Knowledge graphs with interconnected entities and relationships
- **Benefits**: GraphQL queries, relationship traversal, graph analytics
- **Best For**: Enterprise knowledge bases, research databases, social networks

### 3. Flexible Schema Applications
- **Scenario**: Rapidly evolving data structures and content types
- **Benefits**: Dynamic schema inference, automatic property addition
- **Best For**: Startups, experimental platforms, content aggregation systems

### 4. Enterprise Search Platforms
- **Scenario**: Large-scale enterprise search with complex filtering requirements
- **Benefits**: Advanced filtering, role-based access, enterprise security
- **Best For**: Corporate intranets, document management, compliance systems

## Performance Characteristics

### Search Types Performance

| Search Type | Use Case | Speed | Flexibility | Accuracy |
|-------------|----------|-------|-------------|----------|
| **Vector** | Semantic similarity | Fast | Medium | High |
| **Hybrid** | Combined semantic + keyword | Medium | High | Very High |
| **GraphQL** | Complex relationships | Variable | Very High | Perfect |
| **Multi-modal** | Cross-media search | Medium | Very High | High |

### Scaling and Deployment
- **Serverless**: Automatic scaling based on query load
- **Global**: Multi-region deployment for low latency
- **Multi-tenant**: Namespace isolation and access control
- **Performance**: Sub-100ms queries with proper indexing

## Best Practices

1. **Schema Design**: Plan class structure and property types upfront
2. **Vectorization Strategy**: Choose between built-in and external embeddings
3. **Query Optimization**: Use appropriate search types for different use cases
4. **Filtering Strategy**: Create indexed properties for frequent filters
5. **Batch Operations**: Use batch import for large datasets
6. **Monitoring**: Implement query performance monitoring
7. **Security**: Configure proper authentication and authorization
8. **Multi-modal**: Leverage native multi-modal capabilities when applicable

This comprehensive guide provides the foundation for integrating Weaviate Cloud with Swarms agents for sophisticated, multi-modal RAG applications using both built-in and LiteLLM embeddings approaches.