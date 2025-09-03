# SingleStore RAG Integration with Swarms

## Overview

SingleStore is a distributed SQL database with native vector capabilities, combining the power of traditional relational operations with modern vector search functionality. It offers a unique approach to RAG by enabling complex queries that combine structured data, full-text search, and vector similarity in a single, high-performance system. SingleStore is ideal for applications requiring real-time analytics, complex data relationships, and high-throughput vector operations within a familiar SQL interface.

## Key Features

- **Unified SQL + Vector**: Combine relational queries with vector similarity search
- **Real-time Analytics**: Millisecond query performance on streaming data
- **Distributed Architecture**: Horizontal scaling across multiple nodes
- **HTAP Capabilities**: Hybrid transactional and analytical processing
- **Full-text Search**: Built-in text search with ranking and filtering
- **JSON Support**: Native JSON operations and indexing
- **High Throughput**: Handle millions of operations per second
- **Standard SQL**: Familiar SQL interface with vector extensions

## Architecture

SingleStore integrates with Swarms agents as a unified data platform combining vectors with structured data:

```
[Agent] -> [SingleStore Memory] -> [SQL + Vector Engine] -> [Hybrid Results] -> [Enriched Context]
```

The system enables complex queries combining vector similarity with traditional SQL operations for comprehensive data retrieval.

## Setup & Configuration

### Installation

```bash
pip install singlestoredb
pip install swarms
pip install litellm
```

### Environment Variables

```bash
# SingleStore connection
export SINGLESTORE_HOST="your-cluster.singlestore.com"
export SINGLESTORE_PORT="3306"
export SINGLESTORE_USER="your-username"
export SINGLESTORE_PASSWORD="your-password"
export SINGLESTORE_DATABASE="rag_database"

# Optional: SSL configuration
export SINGLESTORE_SSL_DISABLED="false"

# OpenAI API key for LLM
export OPENAI_API_KEY="your-openai-api-key"
```

### Dependencies

- `singlestoredb>=1.0.0`
- `swarms`
- `litellm`
- `numpy`
- `pandas` (for data manipulation)

## Code Example

```python
"""
SingleStore RAG Integration with Swarms Agent

This example demonstrates how to integrate SingleStore as a unified SQL + vector database
for RAG operations combining structured data with vector similarity search.
"""

import os
import singlestoredb as s2
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from swarms import Agent
from litellm import embedding
import json
from datetime import datetime
import uuid

class SingleStoreRAGMemory:
    """SingleStore-based memory system combining SQL and vector operations"""
    
    def __init__(self, 
                 table_name: str = "swarms_documents",
                 embedding_model: str = "text-embedding-3-small",
                 vector_dimension: int = 1536,
                 create_indexes: bool = True):
        """
        Initialize SingleStore RAG memory system
        
        Args:
            table_name: Name of the documents table
            embedding_model: LiteLLM embedding model name
            vector_dimension: Dimension of vector embeddings
            create_indexes: Whether to create optimized indexes
        """
        self.table_name = table_name
        self.embedding_model = embedding_model
        self.vector_dimension = vector_dimension
        self.create_indexes = create_indexes
        
        # Initialize connection
        self.connection = self._create_connection()
        
        # Create table schema
        self._create_table()
        
        # Create indexes for performance
        if create_indexes:
            self._create_indexes()
    
    def _create_connection(self):
        """Create SingleStore connection"""
        connection_params = {
            "host": os.getenv("SINGLESTORE_HOST"),
            "port": int(os.getenv("SINGLESTORE_PORT", "3306")),
            "user": os.getenv("SINGLESTORE_USER"),
            "password": os.getenv("SINGLESTORE_PASSWORD"),
            "database": os.getenv("SINGLESTORE_DATABASE"),
        }
        
        # Optional SSL configuration
        if os.getenv("SINGLESTORE_SSL_DISABLED", "false").lower() != "true":
            connection_params["ssl_disabled"] = False
        
        # Remove None values
        connection_params = {k: v for k, v in connection_params.items() if v is not None}
        
        try:
            conn = s2.connect(**connection_params)
            print(f"Connected to SingleStore: {connection_params['host']}")
            return conn
        except Exception as e:
            raise ConnectionError(f"Failed to connect to SingleStore: {e}")
    
    def _create_table(self):
        """Create documents table with vector and metadata columns"""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id VARCHAR(255) PRIMARY KEY,
            title VARCHAR(1000),
            content TEXT,
            embedding BLOB,
            category VARCHAR(100),
            author VARCHAR(255),
            tags JSON,
            metadata JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            token_count INT,
            content_hash VARCHAR(64),
            
            -- Full-text search index
            FULLTEXT(title, content)
        )
        """
        
        with self.connection.cursor() as cursor:
            cursor.execute(create_table_sql)
        
        print(f"Created/verified table: {self.table_name}")
    
    def _create_indexes(self):
        """Create optimized indexes for performance"""
        indexes = [
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_category ON {self.table_name}(category)",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_author ON {self.table_name}(author)",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_created_at ON {self.table_name}(created_at)",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_token_count ON {self.table_name}(token_count)",
        ]
        
        with self.connection.cursor() as cursor:
            for index_sql in indexes:
                try:
                    cursor.execute(index_sql)
                except Exception as e:
                    print(f"Index creation note: {e}")
        
        print(f"Created indexes for table: {self.table_name}")
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using LiteLLM"""
        response = embedding(
            model=self.embedding_model,
            input=texts
        )
        return [item["embedding"] for item in response["data"]]
    
    def _serialize_embedding(self, embedding_vector: List[float]) -> bytes:
        """Serialize embedding vector for storage"""
        return np.array(embedding_vector, dtype=np.float32).tobytes()
    
    def _deserialize_embedding(self, embedding_bytes: bytes) -> List[float]:
        """Deserialize embedding vector from storage"""
        return np.frombuffer(embedding_bytes, dtype=np.float32).tolist()
    
    def add_documents(self, 
                     documents: List[Dict[str, Any]],
                     batch_size: int = 100) -> List[str]:
        """Add documents with rich metadata to SingleStore"""
        # Generate embeddings for all documents
        texts = [doc.get("content", "") for doc in documents]
        embeddings = self._get_embeddings(texts)
        
        doc_ids = []
        
        # Prepare batch insert
        insert_sql = f"""
        INSERT INTO {self.table_name} 
        (id, title, content, embedding, category, author, tags, metadata, token_count, content_hash)
        VALUES (%(id)s, %(title)s, %(content)s, %(embedding)s, %(category)s, 
                %(author)s, %(tags)s, %(metadata)s, %(token_count)s, %(content_hash)s)
        ON DUPLICATE KEY UPDATE
        title = VALUES(title),
        content = VALUES(content),
        embedding = VALUES(embedding),
        category = VALUES(category),
        author = VALUES(author),
        tags = VALUES(tags),
        metadata = VALUES(metadata),
        updated_at = CURRENT_TIMESTAMP
        """
        
        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            batch_data = []
            for doc, embedding_vec in zip(batch_docs, batch_embeddings):
                doc_id = doc.get("id", str(uuid.uuid4()))
                doc_ids.append(doc_id)
                
                # Calculate content hash for deduplication
                content_hash = str(hash(doc.get("content", "")))
                
                # Estimate token count (rough approximation)
                token_count = len(doc.get("content", "").split()) * 1.3  # Rough token estimate
                
                batch_data.append({
                    "id": doc_id,
                    "title": doc.get("title", ""),
                    "content": doc.get("content", ""),
                    "embedding": self._serialize_embedding(embedding_vec),
                    "category": doc.get("category", ""),
                    "author": doc.get("author", ""),
                    "tags": json.dumps(doc.get("tags", [])),
                    "metadata": json.dumps(doc.get("metadata", {})),
                    "token_count": int(token_count),
                    "content_hash": content_hash
                })
            
            # Execute batch insert
            with self.connection.cursor() as cursor:
                cursor.executemany(insert_sql, batch_data)
        
        self.connection.commit()
        print(f"Added {len(documents)} documents to SingleStore")
        return doc_ids
    
    def vector_search(self, 
                     query: str,
                     limit: int = 5,
                     similarity_threshold: float = 0.7,
                     filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform vector similarity search with optional SQL filters"""
        
        # Generate query embedding
        query_embedding = self._get_embeddings([query])[0]
        query_embedding_bytes = self._serialize_embedding(query_embedding)
        
        # Build base SQL query
        sql_parts = [f"""
        SELECT 
            id, title, content, category, author, tags, metadata, created_at,
            DOT_PRODUCT(embedding, %(query_embedding)s) / 
            (SQRT(DOT_PRODUCT(embedding, embedding)) * SQRT(DOT_PRODUCT(%(query_embedding)s, %(query_embedding)s))) as similarity_score
        FROM {self.table_name}
        WHERE 1=1
        """]
        
        params = {"query_embedding": query_embedding_bytes}
        
        # Add filters
        if filters:
            if "category" in filters:
                sql_parts.append("AND category = %(category)s")
                params["category"] = filters["category"]
            
            if "author" in filters:
                sql_parts.append("AND author = %(author)s")
                params["author"] = filters["author"]
            
            if "date_range" in filters:
                date_range = filters["date_range"]
                if "start" in date_range:
                    sql_parts.append("AND created_at >= %(start_date)s")
                    params["start_date"] = date_range["start"]
                if "end" in date_range:
                    sql_parts.append("AND created_at <= %(end_date)s")
                    params["end_date"] = date_range["end"]
            
            if "tags" in filters:
                sql_parts.append("AND JSON_CONTAINS(tags, %(tags_filter)s)")
                params["tags_filter"] = json.dumps(filters["tags"])
        
        # Add similarity threshold and ordering
        sql_parts.extend([
            f"HAVING similarity_score >= {similarity_threshold}",
            "ORDER BY similarity_score DESC",
            f"LIMIT {limit}"
        ])
        
        final_sql = " ".join(sql_parts)
        
        # Execute query
        with self.connection.cursor() as cursor:
            cursor.execute(final_sql, params)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
        
        # Format results
        formatted_results = {
            "documents": [],
            "metadata": [],
            "scores": [],
            "ids": []
        }
        
        for row in results:
            row_dict = dict(zip(columns, row))
            
            formatted_results["documents"].append(row_dict["content"])
            formatted_results["scores"].append(float(row_dict["similarity_score"]))
            formatted_results["ids"].append(row_dict["id"])
            
            # Parse JSON fields and combine metadata
            tags = json.loads(row_dict["tags"]) if row_dict["tags"] else []
            metadata = json.loads(row_dict["metadata"]) if row_dict["metadata"] else {}
            
            combined_metadata = {
                "title": row_dict["title"],
                "category": row_dict["category"],
                "author": row_dict["author"],
                "tags": tags,
                "created_at": str(row_dict["created_at"]),
                **metadata
            }
            formatted_results["metadata"].append(combined_metadata)
        
        return formatted_results
    
    def hybrid_search(self, 
                     query: str,
                     limit: int = 5,
                     vector_weight: float = 0.7,
                     text_weight: float = 0.3,
                     filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform hybrid search combining vector similarity and full-text search"""
        
        # Generate query embedding
        query_embedding = self._get_embeddings([query])[0]
        query_embedding_bytes = self._serialize_embedding(query_embedding)
        
        # Build hybrid search SQL
        sql_parts = [f"""
        SELECT 
            id, title, content, category, author, tags, metadata, created_at,
            DOT_PRODUCT(embedding, %(query_embedding)s) / 
            (SQRT(DOT_PRODUCT(embedding, embedding)) * SQRT(DOT_PRODUCT(%(query_embedding)s, %(query_embedding)s))) as vector_score,
            MATCH(title, content) AGAINST (%(query_text)s IN NATURAL LANGUAGE MODE) as text_score,
            ({vector_weight} * DOT_PRODUCT(embedding, %(query_embedding)s) / 
             (SQRT(DOT_PRODUCT(embedding, embedding)) * SQRT(DOT_PRODUCT(%(query_embedding)s, %(query_embedding)s))) + 
             {text_weight} * MATCH(title, content) AGAINST (%(query_text)s IN NATURAL LANGUAGE MODE)) as hybrid_score
        FROM {self.table_name}
        WHERE 1=1
        """]
        
        params = {
            "query_embedding": query_embedding_bytes,
            "query_text": query
        }
        
        # Add filters (same as vector_search)
        if filters:
            if "category" in filters:
                sql_parts.append("AND category = %(category)s")
                params["category"] = filters["category"]
            # Add other filters as needed...
        
        # Complete query
        sql_parts.extend([
            "ORDER BY hybrid_score DESC",
            f"LIMIT {limit}"
        ])
        
        final_sql = " ".join(sql_parts)
        
        # Execute and format results (similar to vector_search)
        with self.connection.cursor() as cursor:
            cursor.execute(final_sql, params)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
        
        # Format results with hybrid scoring
        formatted_results = {
            "documents": [],
            "metadata": [],
            "scores": [],
            "vector_scores": [],
            "text_scores": [],
            "ids": []
        }
        
        for row in results:
            row_dict = dict(zip(columns, row))
            
            formatted_results["documents"].append(row_dict["content"])
            formatted_results["scores"].append(float(row_dict["hybrid_score"]))
            formatted_results["vector_scores"].append(float(row_dict["vector_score"]))
            formatted_results["text_scores"].append(float(row_dict["text_score"]))
            formatted_results["ids"].append(row_dict["id"])
            
            # Parse and combine metadata
            tags = json.loads(row_dict["tags"]) if row_dict["tags"] else []
            metadata = json.loads(row_dict["metadata"]) if row_dict["metadata"] else {}
            
            combined_metadata = {
                "title": row_dict["title"],
                "category": row_dict["category"],
                "author": row_dict["author"],
                "tags": tags,
                "created_at": str(row_dict["created_at"]),
                **metadata
            }
            formatted_results["metadata"].append(combined_metadata)
        
        return formatted_results
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get analytics about the document collection"""
        analytics_sql = f"""
        SELECT 
            COUNT(*) as total_documents,
            COUNT(DISTINCT category) as unique_categories,
            COUNT(DISTINCT author) as unique_authors,
            AVG(token_count) as avg_token_count,
            MAX(token_count) as max_token_count,
            MIN(created_at) as earliest_document,
            MAX(created_at) as latest_document
        FROM {self.table_name}
        """
        
        with self.connection.cursor() as cursor:
            cursor.execute(analytics_sql)
            result = cursor.fetchone()
            columns = [desc[0] for desc in cursor.description]
        
        return dict(zip(columns, result)) if result else {}

# Initialize SingleStore RAG memory
memory = SingleStoreRAGMemory(
    table_name="swarms_rag_docs",
    embedding_model="text-embedding-3-small",
    vector_dimension=1536
)

# Sample documents with rich structured data
documents = [
    {
        "title": "SingleStore Vector Capabilities",
        "content": "SingleStore combines SQL databases with native vector search, enabling complex queries that join structured data with similarity search results.",
        "category": "database",
        "author": "Technical Team",
        "tags": ["singlestore", "vectors", "sql"],
        "metadata": {"difficulty": "intermediate", "topic": "hybrid_database"}
    },
    {
        "title": "Real-time Analytics with Vectors",
        "content": "SingleStore's HTAP architecture enables real-time analytics on streaming data while performing vector similarity searches simultaneously.",
        "category": "analytics",
        "author": "Data Scientist",
        "tags": ["analytics", "real-time", "htap"],
        "metadata": {"difficulty": "advanced", "topic": "streaming"}
    },
    {
        "title": "SQL + Vector Integration",
        "content": "The power of SingleStore lies in its ability to combine traditional SQL operations with modern vector search in a single query.",
        "category": "integration",
        "author": "System Architect",
        "tags": ["sql", "integration", "unified"],
        "metadata": {"difficulty": "intermediate", "topic": "sql_vectors"}
    }
]

# Add documents to SingleStore
print("Adding documents to SingleStore...")
doc_ids = memory.add_documents(documents)
print(f"Successfully added {len(doc_ids)} documents")

# Display analytics
analytics = memory.get_analytics()
print(f"Collection analytics: {analytics}")

# Create Swarms agent
agent = Agent(
    agent_name="SingleStore-RAG-Agent",
    agent_description="Advanced agent with SingleStore hybrid SQL + vector RAG capabilities",
    model_name="gpt-4o",
    max_loops=1,
    dynamic_temperature_enabled=True,
)

def query_with_singlestore_rag(query_text: str, 
                              search_type: str = "vector",
                              limit: int = 3,
                              filters: Dict[str, Any] = None):
    """Query with SingleStore RAG using vector, text, or hybrid search"""
    print(f"\nSingleStore {search_type.title()} Search: {query_text}")
    
    # Perform search based on type
    if search_type == "hybrid":
        results = memory.hybrid_search(
            query=query_text,
            limit=limit,
            vector_weight=0.7,
            text_weight=0.3,
            filters=filters
        )
    else:  # vector search
        results = memory.vector_search(
            query=query_text,
            limit=limit,
            filters=filters
        )
    
    if not results["documents"]:
        print("No relevant documents found")
        return agent.run(query_text)
    
    # Prepare enhanced context with metadata
    context_parts = []
    for i, (doc, meta, score) in enumerate(zip(
        results["documents"], 
        results["metadata"], 
        results["scores"]
    )):
        metadata_info = f"[Title: {meta.get('title', 'N/A')}, Category: {meta.get('category', 'N/A')}, Author: {meta.get('author', 'N/A')}]"
        score_info = f"[Score: {score:.3f}"
        
        # Add hybrid score details if available
        if "vector_scores" in results:
            score_info += f", Vector: {results['vector_scores'][i]:.3f}, Text: {results['text_scores'][i]:.3f}"
        score_info += "]"
        
        context_parts.append(f"Document {i+1} {metadata_info} {score_info}:\n{doc}")
    
    context = "\n\n".join(context_parts)
    
    # Enhanced prompt with structured context
    enhanced_prompt = f"""
Based on the following retrieved documents from our SingleStore knowledge base:

{context}

Question: {query_text}

Instructions:
1. Use the structured metadata to understand document context and authority
2. Consider the similarity scores when weighing information importance
3. Provide a comprehensive answer based on the retrieved context
4. Mention specific document titles or authors when referencing information

Response:
"""
    
    return agent.run(enhanced_prompt)

# Example usage
if __name__ == "__main__":
    # Test vector search
    print("=== Vector Search ===")
    response = query_with_singlestore_rag(
        "How does SingleStore combine SQL with vector search?",
        search_type="vector",
        limit=3
    )
    print(f"Answer: {response}\n")
    
    # Test hybrid search
    print("=== Hybrid Search ===")
    response = query_with_singlestore_rag(
        "real-time analytics capabilities",
        search_type="hybrid", 
        limit=2
    )
    print(f"Hybrid Answer: {response}\n")
    
    # Test filtered search
    print("=== Filtered Search ===")
    response = query_with_singlestore_rag(
        "database integration patterns",
        search_type="vector",
        filters={"category": "database"},
        limit=2
    )
    print(f"Filtered Answer: {response}\n")
    
    print("SingleStore RAG integration demonstration completed!")
```

## Use Cases

### 1. **Enterprise Data Platforms**
- Combining operational data with knowledge bases
- Real-time analytics with contextual information
- Customer 360 views with vector similarity

### 2. **Financial Services**
- Risk analysis with document similarity
- Regulatory compliance with structured queries
- Fraud detection combining patterns and text

### 3. **E-commerce Platforms**
- Product recommendations with inventory data
- Customer support with order history
- Content personalization with user behavior

### 4. **Healthcare Systems**
- Patient records with research literature
- Drug discovery with clinical trial data
- Medical imaging with diagnostic text

## Performance Characteristics

### Query Performance
- **Vector Search**: < 10ms for millions of vectors
- **Hybrid Queries**: < 50ms combining SQL + vectors
- **Complex Joins**: Sub-second for structured + vector data
- **Real-time Ingestion**: 100K+ inserts per second

### Scaling Capabilities
- **Distributed**: Linear scaling across cluster nodes
- **Memory**: In-memory processing for hot data
- **Storage**: Tiered storage for cost optimization
- **Concurrency**: Thousands of concurrent queries

## Best Practices

1. **Schema Design**: Optimize table structure for query patterns
2. **Index Strategy**: Create appropriate indexes for filters and joins
3. **Vector Dimensions**: Choose optimal embedding dimensions for your use case
4. **Batch Processing**: Use batch operations for bulk data operations
5. **Query Optimization**: Leverage SQL query optimization techniques
6. **Memory Management**: Configure memory settings for optimal performance
7. **Monitoring**: Use SingleStore's built-in monitoring and metrics
8. **Security**: Implement proper authentication and access controls

This comprehensive guide provides everything needed to integrate SingleStore with Swarms agents for hybrid SQL + vector RAG applications, leveraging the power of unified data processing with the LiteLLM embeddings approach.