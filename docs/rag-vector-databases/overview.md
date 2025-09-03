# RAG Vector Databases

## Overview

This section provides comprehensive guides for integrating various vector databases with Swarms agents for Retrieval-Augmented Generation (RAG) operations. Each guide demonstrates how to use unified LiteLLM embeddings with different vector database systems to create powerful, context-aware AI agents.

## Available Vector Database Integrations

### Cloud-Based Solutions

- **[Pinecone](pinecone.md)** - Serverless vector database with auto-scaling and high availability
- **[Weaviate Cloud](weaviate-cloud.md)** - Multi-modal vector database with GraphQL API
- **[Milvus Cloud](milvus-cloud.md)** - Enterprise-grade managed vector database service

### Self-Hosted Solutions

- **[Qdrant](qdrant.md)** - High-performance vector similarity search engine
- **[ChromaDB](chromadb.md)** - Simple, fast vector database for AI applications
- **[FAISS](faiss.md)** - Facebook's efficient similarity search library
- **[Weaviate Local](weaviate-local.md)** - Self-hosted Weaviate with full control
- **[Milvus Local](milvus-local.md)** - Local Milvus deployment for development

### Specialized Solutions

- **[SingleStore](singlestore.md)** - SQL + Vector hybrid database for complex queries
- **[Zyphra RAG](zyphra-rag.md)** - Specialized RAG system with advanced features

## Key Features Across All Integrations

### Unified LiteLLM Embeddings
All guides use the standardized LiteLLM approach with `text-embedding-3-small` for consistent embedding generation across different vector databases.

### Swarms Agent Integration
Each integration demonstrates how to:
- Initialize vector database connections
- Add documents with rich metadata
- Perform semantic search queries
- Integrate with Swarms agents for RAG operations

### Common Capabilities
- **Semantic Search**: Vector similarity matching for relevant document retrieval
- **Metadata Filtering**: Advanced filtering based on document properties
- **Batch Operations**: Efficient bulk document processing
- **Real-time Updates**: Dynamic knowledge base management
- **Scalability**: Solutions for different scale requirements

## Choosing the Right Vector Database

### For Development & Prototyping
- **ChromaDB**: Simple setup, good for experimentation
- **FAISS**: High performance, good for research
- **Milvus Local**: Feature-rich local development

### For Production Cloud Deployments
- **Pinecone**: Serverless, auto-scaling, managed
- **Weaviate Cloud**: Multi-modal, GraphQL API
- **Milvus Cloud**: Enterprise features, high availability

### For Self-Hosted Production
- **Qdrant**: High performance, clustering support
- **Weaviate Local**: Full control, custom configurations
- **SingleStore**: SQL + Vector hybrid capabilities

### For Specialized Use Cases
- **SingleStore**: When you need both SQL and vector operations
- **Zyphra RAG**: For advanced RAG-specific features
- **FAISS**: When maximum search performance is critical

## Getting Started

1. Choose a vector database based on your requirements
2. Follow the specific integration guide
3. Install required dependencies
4. Configure embeddings with LiteLLM
5. Initialize your Swarms agent with the vector database memory
6. Add your documents and start querying

Each guide provides complete code examples, setup instructions, and best practices for production deployment.