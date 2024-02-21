# `PineconeDB` Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [PineconeVector Class](#pineconevector-class)
3. [Installation](#installation)
4. [Usage](#usage)
   - [Creating a PineconeVector Instance](#creating-a-pineconevector-instance)
   - [Creating an Index](#creating-an-index)
   - [Upserting Vectors](#upserting-vectors)
   - [Querying the Index](#querying-the-index)
   - [Loading an Entry](#loading-an-entry)
   - [Loading Entries](#loading-entries)
5. [Additional Information](#additional-information)
6. [References and Resources](#references-and-resources)

---

## 1. Introduction <a name="introduction"></a>

Welcome to the Swarms documentation! Swarms is a library that provides various memory and storage options for high-dimensional vectors. In this documentation, we will focus on the `PineconeVector` class, which is a vector storage driver that uses Pinecone as the underlying storage engine.

### 1.1 Purpose

The `PineconeVector` class allows you to interact with Pinecone, a vector database that enables the storage, search, and retrieval of high-dimensional vectors with speed and low latency. By using Swarms with Pinecone, you can easily manage and work with vector data in your applications without the need to manage infrastructure.

### 1.2 Key Features

- Seamless integration with Pinecone for vector storage.
- Simple and convenient API for upserting vectors, querying, and loading entries.
- Support for creating and managing indexes.

---

## 2. PineconeVector Class <a name="pineconevector-class"></a>

The `PineconeVector` class is the core component of Swarms that interacts with Pinecone for vector storage. Below, we will provide an in-depth overview of this class, including its purpose, parameters, and methods.

### 2.1 Class Definition

```python
class PineconeVector(BaseVector):
```

### 2.2 Parameters

The `PineconeVector` class accepts the following parameters during initialization:

- `api_key` (str): The API key for your Pinecone account.
- `index_name` (str): The name of the index to use.
- `environment` (str): The environment to use. Either "us-west1-gcp" or "us-east1-gcp".
- `project_name` (str, optional): The name of the project to use. Defaults to `None`.
- `index` (pinecone.Index, optional): The Pinecone index to use. Defaults to `None`.

### 2.3 Methods

The `PineconeVector` class provides several methods for interacting with Pinecone:

#### 2.3.1 `upsert_vector`

```python
def upsert_vector(
    self,
    vector: list[float],
    vector_id: Optional[str] = None,
    namespace: Optional[str] = None,
    meta: Optional[dict] = None,
    **kwargs
) -> str:
```

Upserts a vector into the index.

- `vector` (list[float]): The vector to upsert.
- `vector_id` (Optional[str]): An optional ID for the vector. If not provided, a unique ID will be generated.
- `namespace` (Optional[str]): An optional namespace for the vector.
- `meta` (Optional[dict]): An optional metadata dictionary associated with the vector.
- `**kwargs`: Additional keyword arguments.

#### 2.3.2 `load_entry`

```python
def load_entry(
    self, vector_id: str, namespace: Optional[str] = None
) -> Optional[BaseVector.Entry]:
```

Loads a single vector from the index.

- `vector_id` (str): The ID of the vector to load.
- `namespace` (Optional[str]): An optional namespace for the vector.

#### 2.3.3 `load_entries`

```python
def load_entries(self, namespace: Optional[str] = None) -> list[BaseVector.Entry]:
```

Loads all vectors from the index.

- `namespace` (Optional[str]): An optional namespace for the vectors.

#### 2.3.4 `query`

```python
def query(
    self,
    query: str,
    count: Optional[int] = None,
    namespace: Optional[str] = None,
    include_vectors: bool = False,
    include_metadata=True,
    **kwargs
) -> list[BaseVector.QueryResult]:
```

Queries the index for vectors similar to the given query string.

- `query` (str): The query string.
- `count` (Optional[int]): The maximum number of results to return. If not provided, a default value is used.
- `namespace` (Optional[str]): An optional namespace for the query.
- `include_vectors` (bool): Whether to include vectors in the query results.
- `include_metadata` (bool): Whether to include metadata in the query results.
- `**kwargs`: Additional keyword arguments.

#### 2.3.5 `create_index`

```python
def create_index(self, name: str, **kwargs) -> None:
```

Creates a new index.

- `name` (str): The name of the index to create.
- `**kwargs`: Additional keyword arguments.

---

## 3. Installation <a name="installation"></a>

To use the Swarms library and the `PineconeVector` class, you will need to install the library and its dependencies. Follow these steps to get started:

1. Install Swarms:

```bash
pip install swarms
```

2. Install Pinecone:

You will also need a Pinecone account and API key. Follow the instructions on the Pinecone website to create an account and obtain an API key.

3. Import the necessary modules in your Python code:

```python
from swarms.memory.vector_stores.pinecone import PineconeVector
```

Now you're ready to use the `PineconeVector` class to work with Pinecone for vector storage.

---

## 4. Usage <a name="usage"></a>

In this section, we will provide detailed examples of how to use the `PineconeVector` class for vector storage with Pinecone.

### 4.1 Creating a PineconeVector Instance <a name="creating-a-pineconevector-instance"></a>

To get started, you need to create an instance of the `PineconeVector` class. You will need your Pinecone API key, the name of the index you want to use, and the environment. You can also specify an optional project name if you have one.

```python
pv = PineconeVector(
    api_key="your-api-key",
    index_name="your-index-name",
    environment="us-west1-gcp",
    project_name="your-project-name",
)
```

### 4.2 Creating an Index <a name="creating-an-index"></a>

Before you can upsert vectors, you need to create an index in Pinecone. You can use the `create_index` method for this purpose.

```python
pv.create_index("your-index-name")
```

### 4.3 Upserting Vectors <a name="upserting-vectors"></a>

You can upsert vectors into the Pine

cone index using the `upsert_vector` method. This method allows you to specify the vector, an optional vector ID, namespace, and metadata.

```python
vector = [0.1, 0.2, 0.3, 0.4]
vector_id = "unique-vector-id"
namespace = "your-namespace"
meta = {"key1": "value1", "key2": "value2"}

pv.upsert_vector(vector=vector, vector_id=vector_id, namespace=namespace, meta=meta)
```

### 4.4 Querying the Index <a name="querying-the-index"></a>

You can query the Pinecone index to find vectors similar to a given query string using the `query` method. You can specify the query string, the maximum number of results to return, and other options.

```python
query_string = "your-query-string"
count = 10  # Maximum number of results to return
namespace = "your-namespace"
include_vectors = False  # Set to True to include vectors in results
include_metadata = True

results = pv.query(
    query=query_string,
    count=count,
    namespace=namespace,
    include_vectors=include_vectors,
    include_metadata=include_metadata,
)

# Process the query results
for result in results:
    vector_id = result.id
    vector = result.vector
    score = result.score
    meta = result.meta

    # Handle the results as needed
```

### 4.5 Loading an Entry <a name="loading-an-entry"></a>

You can load a single vector entry from the Pinecone index using the `load_entry` method. Provide the vector ID and an optional namespace.

```python
vector_id = "your-vector-id"
namespace = "your-namespace"

entry = pv.load_entry(vector_id=vector_id, namespace=namespace)

if entry is not None:
    loaded_vector = entry.vector
    loaded_meta = entry.meta

    # Use the loaded vector and metadata
else:
    # Vector not found
```

### 4.6 Loading Entries <a name="loading-entries"></a>

To load all vectors from the Pinecone index, you can use the `load_entries` method. You can also specify an optional namespace.

```python
namespace = "your-namespace"

entries = pv.load_entries(namespace=namespace)

# Process the loaded entries
for entry in entries:
    vector_id = entry.id
    vector = entry.vector
    meta = entry.meta

    # Handle the loaded entries as needed
```

---

## 5. Additional Information <a name="additional-information"></a>

In this section, we provide additional information and tips for using the `PineconeVector` class effectively.

- When upserting vectors, you can generate a unique vector ID using a hash of the vector's content to ensure uniqueness.
- Consider using namespaces to organize and categorize vectors within your Pinecone index.
- Pinecone provides powerful querying capabilities, so be sure to explore and leverage its features to retrieve relevant vectors efficiently.
- Keep your Pinecone API key secure and follow Pinecone's best practices for API key management.

---

## 6. References and Resources <a name="references-and-resources"></a>

Here are some references and resources for further information on Pinecone and Swarms:

- [Pinecone Website](https://www.pinecone.io/): Official Pinecone website for documentation and resources.
- [Pinecone Documentation](https://docs.pinecone.io/): Detailed documentation for Pinecone.
- [Swarms GitHub Repository](https://github.com/swarms): Swarms library on GitHub for updates and contributions.

---

This concludes the documentation for the Swarms library and the `PineconeVector` class. You now have a deep understanding of how to use Swarms with Pinecone for vector storage. If you have any further questions or need assistance, please refer to the provided references and resources. Happy coding!