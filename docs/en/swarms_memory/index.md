# Announcing the Release of Swarms-Memory Package: Your Gateway to Efficient RAG Systems


We are thrilled to announce the release of the Swarms-Memory package, a powerful and easy-to-use toolkit designed to facilitate the implementation of Retrieval-Augmented Generation (RAG) systems. Whether you're a seasoned AI practitioner or just starting out, Swarms-Memory provides the tools you need to integrate high-performance, reliable RAG systems into your applications seamlessly.

In this blog post, we'll walk you through getting started with the Swarms-Memory package, covering installation, usage examples, and a detailed overview of supported RAG systems like Pinecone and ChromaDB. Let's dive in!

## What is Swarms-Memory?

Swarms-Memory is a Python package that simplifies the integration of advanced RAG systems into your projects. It supports multiple databases optimized for AI tasks, providing you with the flexibility to choose the best system for your needs. With Swarms-Memory, you can effortlessly handle large-scale AI tasks, vector searches, and more.

### Key Features

- **Easy Integration**: Quickly set up and start using powerful RAG systems.
- **Customizable**: Define custom embedding, preprocessing, and postprocessing functions.
- **Flexible**: Supports multiple RAG systems like ChromaDB and Pinecone, with more coming soon.
- **Scalable**: Designed to handle large-scale AI tasks efficiently.

## Supported RAG Systems

Here's an overview of the RAG systems currently supported by Swarms-Memory:

| RAG System | Status       | Description                                                                              | Documentation             | Website         |
|------------|--------------|------------------------------------------------------------------------------------------|---------------------------|-----------------|
| ChromaDB   | Available    | A high-performance, distributed database optimized for handling large-scale AI tasks.    | [ChromaDB Documentation](https://chromadb.com/docs) | [ChromaDB](https://chromadb.com) |
| Pinecone   | Available    | A fully managed vector database for adding vector search to your applications.           | [Pinecone Documentation](https://pinecone.io/docs) | [Pinecone](https://pinecone.io) |
| Redis      | Coming Soon  | An open-source, in-memory data structure store, used as a database, cache, and broker.   | [Redis Documentation](https://redis.io/documentation) | [Redis](https://redis.io) |
| Faiss      | Coming Soon  | A library for efficient similarity search and clustering of dense vectors by Facebook AI. | [Faiss Documentation](https://faiss.ai) | [Faiss](https://faiss.ai) |
| HNSW       | Coming Soon  | A graph-based algorithm for approximate nearest neighbor search, known for speed.        | [HNSW Documentation](https://hnswlib.github.io/hnswlib) | [HNSW](https://hnswlib.github.io/hnswlib) |

## Getting Started

### Requirements

Before you begin, ensure you have the following:

- Python 3.10
- `.env` file with your respective API keys (e.g., `PINECONE_API_KEY`)

### Installation

You can install the Swarms-Memory package using pip:

```bash
$ pip install swarms-memory
```

### Usage Examples

#### Pinecone

Here's a step-by-step guide on how to use Pinecone with Swarms-Memory:

1. **Import Required Libraries**:

```python
from typing import List, Dict, Any
from swarms_memory import PineconeMemory
```

2. **Define Custom Functions**:

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Custom embedding function using a HuggingFace model
def custom_embedding_function(text: str) -> List[float]:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return embeddings

# Custom preprocessing function
def custom_preprocess(text: str) -> str:
    return text.lower().strip()

# Custom postprocessing function
def custom_postprocess(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for result in results:
        result["custom_score"] = result["score"] * 2  # Example modification
    return results
```

3. **Initialize the Wrapper with Custom Functions**:

```python
wrapper = PineconeMemory(
    api_key="your-api-key",
    environment="your-environment",
    index_name="your-index-name",
    embedding_function=custom_embedding_function,
    preprocess_function=custom_preprocess,
    postprocess_function=custom_postprocess,
    logger_config={
        "handlers": [
            {"sink": "custom_rag_wrapper.log", "rotation": "1 GB"},
            {"sink": lambda msg: print(f"Custom log: {msg}", end="")},
        ],
    },
)
```

4. **Add Documents and Query**:

```python
# Adding documents
wrapper.add("This is a sample document about artificial intelligence.", {"category": "AI"})
wrapper.add("Python is a popular programming language for data science.", {"category": "Programming"})

# Querying
results = wrapper.query("What is AI?", filter={"category": "AI"})
for result in results:
    print(f"Score: {result['score']}, Custom Score: {result['custom_score']}, Text: {result['metadata']['text']}")
```

#### ChromaDB

Using ChromaDB with Swarms-Memory is straightforward. Here’s how:

1. **Import ChromaDB**:

```python
from swarms_memory import ChromaDB
```

2. **Initialize ChromaDB**:

```python
chromadb = ChromaDB(
    metric="cosine",
    output_dir="results",
    limit_tokens=1000,
    n_results=2,
    docs_folder="path/to/docs",
    verbose=True,
)
```

3. **Add and Query Documents**:

```python
# Add a document
doc_id = chromadb.add("This is a test document.")

# Query the document
result = chromadb.query("This is a test query.")

# Traverse a directory
chromadb.traverse_directory()

# Display the result
print(result)
```

## Join the Community

We're excited to see how you leverage Swarms-Memory in your projects! Join our community on Discord to share your experiences, ask questions, and stay updated on the latest developments.

- **🐦 Twitter**: [Follow us on Twitter](https://twitter.com/swarms_platform)
- **📢 Discord**: [Join the Agora Discord](https://discord.gg/agora)
- **Swarms Platform**: [Visit our website](https://swarms.ai)
- **📙 Documentation**: [Read the Docs](https://docs.swarms.ai)

## Conclusion

The Swarms-Memory package brings a new level of ease and efficiency to building and managing RAG systems. With support for leading databases like ChromaDB and Pinecone, it's never been easier to integrate powerful, scalable AI solutions into your projects. We can't wait to see what you'll create with Swarms-Memory!

For more detailed usage examples and documentation, visit our [GitHub repository](https://github.com/swarms-ai/swarms-memory) and start exploring today!
