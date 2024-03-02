# Qdrant Client Library

## Overview

The Qdrant Client Library is designed for interacting with the Qdrant vector database, allowing efficient storage and retrieval of high-dimensional vector data. It integrates with machine learning models for embedding and is particularly suited for search and recommendation systems.

## Installation

```python
pip install qdrant-client sentence-transformers httpx
```

## Class Definition: Qdrant

```python
class Qdrant:
    def __init__(
        self,
        api_key: str,
        host: str,
        port: int = 6333,
        collection_name: str = "qdrant",
        model_name: str = "BAAI/bge-small-en-v1.5",
        https: bool = True,
    ):
        ...
```

### Constructor Parameters

| Parameter       | Type    | Description                                      | Default Value         |
|-----------------|---------|--------------------------------------------------|-----------------------|
| api_key         | str     | API key for authentication.                      | -                     |
| host            | str     | Host address of the Qdrant server.               | -                     |
| port            | int     | Port number for the Qdrant server.               | 6333                  |
| collection_name | str     | Name of the collection to be used or created.    | "qdrant"              |
| model_name      | str     | Name of the sentence transformer model.          | "BAAI/bge-small-en-v1.5" |
| https           | bool    | Flag to use HTTPS for connection.                | True                  |

### Methods

#### `_load_embedding_model(model_name: str)`

Loads the sentence embedding model.

#### `_setup_collection()`

Checks if the specified collection exists in Qdrant; if not, creates it.

#### `add_vectors(docs: List[dict]) -> OperationResponse`

Adds vectors to the Qdrant collection.

#### `search_vectors(query: str, limit: int = 3) -> SearchResult`

Searches the Qdrant collection for vectors similar to the query vector.

## Usage Examples

### Example 1: Setting Up the Qdrant Client

```python
from qdrant_client import Qdrant

qdrant_client = Qdrant(api_key="your_api_key", host="localhost", port=6333)
```

### Example 2: Adding Vectors to a Collection

```python
documents = [{"page_content": "Sample text 1"}, {"page_content": "Sample text 2"}]

operation_info = qdrant_client.add_vectors(documents)
print(operation_info)
```

### Example 3: Searching for Vectors

```python
search_result = qdrant_client.search_vectors("Sample search query")
print(search_result)
```

## Further Information

Refer to the [Qdrant Documentation](https://qdrant.tech/docs) for more details on the Qdrant vector database.
