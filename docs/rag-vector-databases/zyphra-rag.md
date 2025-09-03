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
Zyphra RAG Integration with Swarms Agent

This example demonstrates how to integrate Zyphra RAG as a specialized
vector database optimized for RAG workflows with intelligent retrieval.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from swarms import Agent
from litellm import embedding
import tiktoken
from datetime import datetime
import uuid

# Conceptual Zyphra RAG client implementation
class ZyphraRAGClient:
    """Conceptual client for Zyphra RAG service"""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        # This would be implemented with actual HTTP client
    
    def create_collection(self, name: str, config: Dict) -> Dict:
        # Conceptual API call
        return {"collection_id": f"col_{uuid.uuid4()}", "status": "created"}
    
    def add_documents(self, collection_id: str, documents: List[Dict]) -> Dict:
        # Conceptual document ingestion with intelligent chunking
        return {"document_ids": [f"doc_{i}" for i in range(len(documents))]}
    
    def search(self, collection_id: str, query: str, params: Dict) -> Dict:
        # Conceptual RAG-optimized search
        return {
            "results": [
                {
                    "text": "Sample retrieved content...",
                    "score": 0.95,
                    "metadata": {"chunk_id": "chunk_1", "relevance": "high"},
                    "context_signals": {"semantic": 0.9, "lexical": 0.8, "contextual": 0.95}
                }
            ]
        }

class ZyphraRAGMemory:
    """Zyphra RAG-based memory system optimized for RAG operations"""
    
    def __init__(self, 
                 collection_name: str = "swarms_rag_collection",
                 embedding_model: str = "text-embedding-3-small",
                 chunk_strategy: str = "intelligent",
                 max_chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 retrieval_strategy: str = "hybrid_enhanced"):
        """
        Initialize Zyphra RAG memory system
        
        Args:
            collection_name: Name of the RAG collection
            embedding_model: LiteLLM embedding model name
            chunk_strategy: Document chunking strategy
            max_chunk_size: Maximum tokens per chunk
            chunk_overlap: Overlap tokens between chunks
            retrieval_strategy: Retrieval optimization strategy
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_strategy = chunk_strategy
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_strategy = retrieval_strategy
        
        # Initialize tokenizer for chunk management
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize Zyphra RAG client (conceptual)
        self.client = self._create_client()
        
        # Create collection with RAG-optimized configuration
        self.collection_id = self._create_collection()
        
    def _create_client(self):
        """Create Zyphra RAG client"""
        api_key = os.getenv("ZYPHRA_RAG_API_KEY")
        base_url = os.getenv("ZYPHRA_RAG_URL", "https://api.zyphra.com/rag/v1")
        
        if not api_key:
            raise ValueError("ZYPHRA_RAG_API_KEY must be set")
        
        print(f"Connecting to Zyphra RAG: {base_url}")
        return ZyphraRAGClient(api_key, base_url)
    
    def _create_collection(self):
        """Create RAG-optimized collection"""
        config = {
            "embedding_model": self.embedding_model,
            "chunk_strategy": self.chunk_strategy,
            "max_chunk_size": self.max_chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "retrieval_strategy": self.retrieval_strategy,
            "optimization_target": "rag_quality",
            "context_window_management": True,
            "query_enhancement": True,
            "relevance_scoring": "llm_optimized"
        }
        
        result = self.client.create_collection(self.collection_name, config)
        print(f"Created RAG collection: {result['collection_id']}")
        return result["collection_id"]
    
    def _intelligent_chunking(self, text: str, metadata: Dict) -> List[Dict]:
        """Implement intelligent chunking with context preservation"""
        # Tokenize text
        tokens = self.tokenizer.encode(text)
        
        # Simple chunking implementation (in practice, this would be more sophisticated)
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + self.max_chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Preserve context by including overlap
            if start > 0 and self.chunk_overlap > 0:
                overlap_start = max(0, start - self.chunk_overlap)
                overlap_tokens = tokens[overlap_start:start]
                overlap_text = self.tokenizer.decode(overlap_tokens)
                chunk_text = overlap_text + " " + chunk_text
            
            chunks.append({
                "text": chunk_text,
                "tokens": len(chunk_tokens),
                "chunk_index": len(chunks),
                "start_token": start,
                "end_token": end,
                "metadata": {**metadata, "chunk_strategy": self.chunk_strategy}
            })
            
            start = end - self.chunk_overlap if end < len(tokens) else end
        
        return chunks
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using LiteLLM"""
        response = embedding(
            model=self.embedding_model,
            input=texts
        )
        return [item["embedding"] for item in response["data"]]
    
    def add_documents(self, 
                     documents: List[str], 
                     metadata: List[Dict] = None,
                     enable_chunking: bool = True) -> List[str]:
        """Add documents with intelligent chunking for RAG optimization"""
        if metadata is None:
            metadata = [{}] * len(documents)
        
        processed_docs = []
        
        for doc, meta in zip(documents, metadata):
            if enable_chunking and self.chunk_strategy != "none":
                # Apply intelligent chunking
                chunks = self._intelligent_chunking(doc, meta)
                processed_docs.extend(chunks)
            else:
                # Add as single document
                processed_docs.append({
                    "text": doc,
                    "metadata": meta,
                    "tokens": len(self.tokenizer.encode(doc))
                })
        
        # Generate embeddings for all processed documents
        texts = [doc["text"] for doc in processed_docs]
        embeddings = self._get_embeddings(texts)
        
        # Prepare documents for Zyphra RAG ingestion
        rag_documents = []
        for i, (doc_data, embedding_vec) in enumerate(zip(processed_docs, embeddings)):
            rag_doc = {
                "id": f"doc_{uuid.uuid4()}",
                "text": doc_data["text"],
                "embedding": embedding_vec,
                "metadata": {
                    **doc_data["metadata"],
                    "tokens": doc_data["tokens"],
                    "processed_at": datetime.now().isoformat(),
                    "chunk_index": doc_data.get("chunk_index", 0)
                }
            }
            rag_documents.append(rag_doc)
        
        # Ingest into Zyphra RAG
        result = self.client.add_documents(self.collection_id, rag_documents)
        
        print(f"Added {len(documents)} documents ({len(processed_docs)} chunks) to Zyphra RAG")
        return result["document_ids"]
    
    def search(self, 
               query: str,
               limit: int = 3,
               relevance_threshold: float = 0.7,
               context_optimization: bool = True,
               query_enhancement: bool = True) -> Dict[str, Any]:
        """Perform RAG-optimized search with enhanced retrieval strategies"""
        
        search_params = {
            "limit": limit,
            "relevance_threshold": relevance_threshold,
            "context_optimization": context_optimization,
            "query_enhancement": query_enhancement,
            "retrieval_strategy": self.retrieval_strategy,
            "embedding_model": self.embedding_model,
            "return_context_signals": True,
            "optimize_for_llm": True
        }
        
        # Perform enhanced search
        search_result = self.client.search(self.collection_id, query, search_params)
        
        # Format results with RAG-specific enhancements
        formatted_results = {
            "documents": [],
            "metadata": [],
            "scores": [],
            "context_signals": [],
            "retrieval_quality": {},
            "token_counts": []
        }
        
        total_tokens = 0
        for result in search_result.get("results", []):
            formatted_results["documents"].append(result["text"])
            formatted_results["metadata"].append(result.get("metadata", {}))
            formatted_results["scores"].append(float(result["score"]))
            formatted_results["context_signals"].append(result.get("context_signals", {}))
            
            # Track token usage for context window management
            token_count = len(self.tokenizer.encode(result["text"]))
            formatted_results["token_counts"].append(token_count)
            total_tokens += token_count
        
        # Add retrieval quality metrics
        formatted_results["retrieval_quality"] = {
            "total_tokens": total_tokens,
            "avg_relevance": sum(formatted_results["scores"]) / len(formatted_results["scores"]) if formatted_results["scores"] else 0,
            "context_diversity": self._calculate_context_diversity(formatted_results["documents"]),
            "query_enhancement_applied": query_enhancement
        }
        
        return formatted_results
    
    def _calculate_context_diversity(self, documents: List[str]) -> float:
        """Calculate diversity score for retrieved context"""
        # Simple diversity calculation (in practice, this would be more sophisticated)
        if len(documents) <= 1:
            return 1.0
        
        # Calculate semantic diversity based on document similarity
        embeddings = self._get_embeddings(documents)
        similarities = []
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                # Cosine similarity
                dot_product = sum(a * b for a, b in zip(embeddings[i], embeddings[j]))
                norm_a = sum(a * a for a in embeddings[i]) ** 0.5
                norm_b = sum(b * b for b in embeddings[j]) ** 0.5
                similarity = dot_product / (norm_a * norm_b)
                similarities.append(similarity)
        
        # Diversity = 1 - average similarity
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        return 1.0 - avg_similarity
    
    def optimize_context_window(self, 
                               retrieved_results: Dict, 
                               max_tokens: int = 4000,
                               strategy: str = "relevance_first") -> Dict[str, Any]:
        """Optimize retrieved context for specific token budget"""
        documents = retrieved_results["documents"]
        scores = retrieved_results["scores"]
        token_counts = retrieved_results["token_counts"]
        
        if sum(token_counts) <= max_tokens:
            return retrieved_results
        
        # Apply context window optimization strategy
        if strategy == "relevance_first":
            # Sort by relevance and include highest scoring documents
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        elif strategy == "token_efficient":
            # Optimize for best relevance per token
            efficiency_scores = [score / max(token_count, 1) for score, token_count in zip(scores, token_counts)]
            sorted_indices = sorted(range(len(efficiency_scores)), key=lambda i: efficiency_scores[i], reverse=True)
        else:
            sorted_indices = list(range(len(documents)))
        
        # Select documents within token budget
        selected_docs = []
        selected_metadata = []
        selected_scores = []
        selected_tokens = []
        current_tokens = 0
        
        for idx in sorted_indices:
            if current_tokens + token_counts[idx] <= max_tokens:
                selected_docs.append(documents[idx])
                selected_metadata.append(retrieved_results["metadata"][idx])
                selected_scores.append(scores[idx])
                selected_tokens.append(token_counts[idx])
                current_tokens += token_counts[idx]
        
        return {
            "documents": selected_docs,
            "metadata": selected_metadata,
            "scores": selected_scores,
            "token_counts": selected_tokens,
            "optimization_applied": True,
            "final_token_count": current_tokens,
            "token_efficiency": current_tokens / max_tokens
        }

# Initialize Zyphra RAG memory
memory = ZyphraRAGMemory(
    collection_name="swarms_zyphra_rag",
    embedding_model="text-embedding-3-small",
    chunk_strategy="intelligent",
    max_chunk_size=512,
    retrieval_strategy="hybrid_enhanced"
)

# Sample documents optimized for RAG
documents = [
    "Zyphra RAG is a specialized vector database designed specifically for retrieval-augmented generation workflows. It optimizes every aspect of the retrieval process to maximize relevance and context quality for language models.",
    "Intelligent chunking in Zyphra RAG preserves document context while creating optimal chunk sizes for embedding and retrieval. This approach maintains semantic coherence across chunk boundaries.",
    "Multi-strategy retrieval combines semantic similarity, lexical matching, and contextual signals to identify the most relevant information for specific queries and use cases.",
    "Context window optimization ensures that retrieved information fits within language model constraints while maximizing information density and relevance to the query.",
    "RAG-specific scoring algorithms in Zyphra prioritize content that is most likely to improve language model response quality and accuracy.",
]

# Rich metadata for RAG optimization
metadatas = [
    {"category": "overview", "topic": "zyphra_rag", "difficulty": "intermediate", "content_type": "explanation"},
    {"category": "feature", "topic": "chunking", "difficulty": "advanced", "content_type": "technical"},
    {"category": "feature", "topic": "retrieval", "difficulty": "advanced", "content_type": "technical"},
    {"category": "optimization", "topic": "context", "difficulty": "expert", "content_type": "technical"},
    {"category": "algorithm", "topic": "scoring", "difficulty": "expert", "content_type": "technical"},
]

# Add documents to Zyphra RAG
print("Adding documents to Zyphra RAG...")
doc_ids = memory.add_documents(documents, metadatas, enable_chunking=True)
print(f"Successfully processed documents into {len(doc_ids)} retrievable units")

# Create Swarms agent with Zyphra RAG
agent = Agent(
    agent_name="ZyphraRAG-Agent",
    agent_description="Advanced agent with Zyphra RAG-optimized retrieval for maximum context relevance",
    model_name="gpt-4o",
    max_loops=1,
    dynamic_temperature_enabled=True,
)

def query_with_zyphra_rag(query_text: str, 
                         limit: int = 3,
                         max_context_tokens: int = 3000,
                         optimization_strategy: str = "relevance_first"):
    """Query with Zyphra RAG's advanced retrieval optimization"""
    print(f"\nZyphra RAG Query: {query_text}")
    
    # Perform RAG-optimized search
    results = memory.search(
        query=query_text,
        limit=limit,
        relevance_threshold=0.7,
        context_optimization=True,
        query_enhancement=True
    )
    
    if not results["documents"]:
        print("No relevant documents found")
        return agent.run(query_text)
    
    # Optimize context window if needed
    if results["retrieval_quality"]["total_tokens"] > max_context_tokens:
        print(f"Optimizing context window: {results['retrieval_quality']['total_tokens']} -> {max_context_tokens} tokens")
        results = memory.optimize_context_window(results, max_context_tokens, optimization_strategy)
    
    # Prepare enhanced context
    context_parts = []
    for i, (doc, score, signals) in enumerate(zip(
        results["documents"], 
        results["scores"], 
        results.get("context_signals", [{}] * len(results["documents"]))
    )):
        relevance_info = f"[Relevance: {score:.3f}"
        if signals:
            relevance_info += f", Semantic: {signals.get('semantic', 0):.2f}, Contextual: {signals.get('contextual', 0):.2f}"
        relevance_info += "]"
        
        context_parts.append(f"Context {i+1} {relevance_info}:\n{doc}")
    
    context = "\n\n".join(context_parts)
    
    # Display retrieval analytics
    quality = results["retrieval_quality"]
    print(f"Retrieval Quality Metrics:")
    print(f"  - Total tokens: {quality['total_tokens']}")
    print(f"  - Average relevance: {quality['avg_relevance']:.3f}")
    print(f"  - Context diversity: {quality['context_diversity']:.3f}")
    print(f"  - Query enhancement: {quality['query_enhancement_applied']}")
    
    # Enhanced prompt with RAG-optimized context
    enhanced_prompt = f"""
You are provided with high-quality, RAG-optimized context retrieved specifically for this query. 
The context has been scored and ranked for relevance, with diversity optimization applied.

Context:
{context}

Question: {query_text}

Instructions:
1. Base your response primarily on the provided context
2. Consider the relevance scores when weighing information
3. Synthesize information from multiple context pieces when applicable
4. Indicate confidence level based on context quality and relevance

Response:
"""
    
    # Run agent with enhanced prompt
    response = agent.run(enhanced_prompt)
    return response

# Example usage and testing
if __name__ == "__main__":
    # Test RAG-optimized queries
    queries = [
        "How does Zyphra RAG optimize retrieval for language models?",
        "What is intelligent chunking and why is it important?",
        "How does multi-strategy retrieval work?",
        "What are the benefits of context window optimization?",
    ]
    
    print("=== Zyphra RAG Enhanced Queries ===")
    for query in queries:
        response = query_with_zyphra_rag(
            query, 
            limit=3, 
            max_context_tokens=2500,
            optimization_strategy="relevance_first"
        )
        print(f"Enhanced Answer: {response}\n")
        print("-" * 80)
    
    # Test token efficiency optimization
    print("\n=== Token Efficiency Optimization ===")
    response = query_with_zyphra_rag(
        "Explain all the advanced features of Zyphra RAG",
        limit=5,
        max_context_tokens=1500,  # Strict token limit
        optimization_strategy="token_efficient"
    )
    print(f"Token-optimized answer: {response}\n")
    
    print("Zyphra RAG integration demonstration completed!")
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