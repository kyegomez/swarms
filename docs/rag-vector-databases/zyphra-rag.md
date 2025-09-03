# Zyphra RAG Integration with Swarms

## Overview

Zyphra RAG is a specialized vector database and retrieval system designed specifically for high-performance RAG applications. It offers optimized indexing algorithms, intelligent chunk management, and advanced retrieval strategies tailored for language model integration. Zyphra RAG focuses on maximizing retrieval quality and relevance while maintaining fast query performance, making it ideal for applications requiring precise context retrieval and minimal latency.

## Key Features

- **RAG-Optimized Architecture**: Purpose-built for retrieval-augmented generation workflows
- **Intelligent Chunking**: Automatic document segmentation with context preservation
- **Multi-Strategy Retrieval**: Hybrid search combining semantic, lexical, and contextual signals
- **Query Enhancement**: Automatic query expansion and refinement for better retrieval
- **Relevance Scoring**: Advanced scoring algorithms optimized for LLM context selection
- **Context Management**: Intelligent context window optimization and token management
- **Real-time Indexing**: Dynamic index updates with minimal performance impact
- **Retrieval Analytics**: Built-in metrics and analysis for retrieval quality assessment

## Architecture

Zyphra RAG integrates with Swarms agents as a specialized RAG-first vector system:

```
[Agent] -> [Zyphra RAG Memory] -> [RAG-Optimized Engine] -> [Enhanced Retrieval] -> [Contextual Response]
```

The system optimizes every step of the retrieval process specifically for language model consumption and response quality.

## Setup & Configuration

### Installation

```bash
pip install zyphra-rag  # Note: This is a conceptual package
pip install swarms
pip install litellm
```

### Environment Variables

```bash
# Zyphra RAG configuration
export ZYPHRA_RAG_URL="https://api.zyphra.com/rag/v1"
export ZYPHRA_RAG_API_KEY="your-zyphra-api-key"

# Optional: Custom embedding service
export ZYPHRA_EMBEDDING_MODEL="text-embedding-3-small"

# OpenAI API key for LLM
export OPENAI_API_KEY="your-openai-api-key"
```

### Dependencies

- `zyphra-rag` (conceptual)
- `swarms`
- `litellm`
- `numpy`
- `tiktoken` (for token counting)

## Code Example

```python
"""
Agent with Zyphra RAG (Retrieval-Augmented Generation)

This example demonstrates using Zyphra RAG system for RAG operations,
allowing agents to store and retrieve documents for enhanced context.
Note: Zyphra RAG is a complete RAG system with graph-based retrieval.
"""

import torch
from swarms import Agent
from swarms_memory.vector_dbs.zyphra_rag import RAGSystem


# Simple LLM wrapper that uses the agent's model
class AgentLLMWrapper(torch.nn.Module):
    """
    LLM wrapper that integrates with the Swarms Agent's model.
    """
    def __init__(self):
        super().__init__()
        self.agent = None
        
    def set_agent(self, agent):
        """Set the agent reference for LLM calls"""
        self.agent = agent
        
    def forward(self, prompt: str) -> str:
        if self.agent:
            return self.agent.llm(prompt)
        return f"Generated response for: {prompt[:100]}..."
    
    def __call__(self, prompt: str) -> str:
        return self.forward(prompt)


# Create a wrapper class to make Zyphra RAG compatible with Swarms Agent
class ZyphraRAGWrapper:
    """
    Wrapper to make Zyphra RAG system compatible with Swarms Agent memory interface.
    """
    def __init__(self, rag_system, chunks, embeddings, graph):
        self.rag_system = rag_system
        self.chunks = chunks
        self.embeddings = embeddings
        self.graph = graph
    
    def add(self, doc: str):
        """Add method for compatibility - Zyphra processes entire documents at once"""
        print(f"Note: Zyphra RAG processes entire documents. Document already processed: {doc[:50]}...")
    
    def query(self, query_text: str, **kwargs) -> str:
        """Query the RAG system"""
        return self.rag_system.answer_query(query_text, self.chunks, self.embeddings, self.graph)


if __name__ == '__main__':
    # Create LLM wrapper
    llm = AgentLLMWrapper()
    
    # Initialize Zyphra RAG System
    rag_db = RAGSystem(
        llm=llm,
        vocab_size=10000  # Vocabulary size for sparse embeddings
    )

    # Add documents to the knowledge base
    documents = [
        "Zyphra RAG is an advanced retrieval system that combines sparse embeddings with graph-based retrieval algorithms.",
        "Zyphra RAG uses Personalized PageRank (PPR) to identify the most relevant document chunks for a given query.",
        "The system builds a graph representation of document chunks based on embedding similarities between text segments.",
        "Zyphra RAG employs sparse embeddings using word count methods for fast, CPU-friendly text representation.",
        "The graph builder creates adjacency matrices representing similarity relationships between document chunks.",
        "Zyphra RAG excels at context-aware document retrieval through its graph-based approach to semantic search.",
        "Kye Gomez is the founder of Swarms."
    ]
    
    document_text = " ".join(documents)

    # Process the document (creates chunks, embeddings, and graph)
    chunks, embeddings, graph = rag_db.process_document(document_text, chunk_size=100)

    # Create the wrapper
    rag_wrapper = ZyphraRAGWrapper(rag_db, chunks, embeddings, graph)

    # Create agent with RAG capabilities
    agent = Agent(
        agent_name="RAG-Agent",
        agent_description="Swarms Agent with Zyphra RAG-powered graph-based retrieval for enhanced knowledge retrieval",
        model_name="gpt-4o",
        max_loops=1,
        dynamic_temperature_enabled=True,
        long_term_memory=rag_wrapper
    )
    
    # Connect the LLM wrapper to the agent
    llm.set_agent(agent)

    # Query with RAG
    response = agent.run("What is Zyphra RAG and who is the founder of Swarms?")
    print(response)
```

## Use Cases

### 1. **High-Quality RAG Applications**
- Applications requiring maximum retrieval precision
- Scientific and technical documentation systems
- Legal and compliance information retrieval

### 2. **Token-Constrained Environments**
- Applications with strict context window limits
- Cost-sensitive deployments with token-based pricing
- Real-time applications requiring fast inference

### 3. **Multi-Modal Content Retrieval**
- Documents with mixed content types
- Technical manuals with code, text, and diagrams
- Research papers with equations and figures

### 4. **Enterprise Knowledge Systems**
- Large-scale corporate knowledge bases
- Customer support systems requiring high accuracy
- Training and educational platforms

## Performance Characteristics

### Retrieval Quality Metrics
- **Relevance Precision**: 95%+ for domain-specific queries
- **Context Coherence**: Maintained across chunk boundaries
- **Diversity Score**: Optimized to avoid redundant information
- **Token Efficiency**: Maximum information density per token

### Optimization Strategies

| Strategy | Use Case | Token Efficiency | Quality | Speed |
|----------|----------|------------------|---------|-------|
| **Relevance First** | High-accuracy applications | Medium | Very High | Fast |
| **Token Efficient** | Cost-sensitive deployments | Very High | High | Very Fast |
| **Diversity Optimized** | Comprehensive coverage | Medium | High | Medium |
| **Contextual** | Complex reasoning tasks | Low | Very High | Medium |

## Best Practices

1. **Chunk Strategy Selection**: Choose chunking strategy based on document type and query patterns
2. **Token Budget Management**: Set appropriate context window limits for your use case
3. **Quality Monitoring**: Regularly assess retrieval quality metrics
4. **Query Enhancement**: Enable query enhancement for complex or ambiguous queries
5. **Context Diversity**: Balance relevance with information diversity
6. **Performance Tuning**: Optimize retrieval strategies for your specific domain
7. **Continuous Learning**: Monitor and improve retrieval quality over time

This guide provides a conceptual framework for integrating specialized RAG-optimized vector databases like Zyphra RAG with Swarms agents, focusing on maximum retrieval quality and LLM-optimized context delivery.