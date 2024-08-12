# `PgVectorVectorStore` Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Overview](#overview)
3. [Class Definition](#class-definition)
4. [Functionality and Usage](#functionality-and-usage)
   - [Setting Up the Database](#setting-up-the-database)
   - [Upserting Vectors](#upserting-vectors)
   - [Loading Vector Entries](#loading-vector-entries)
   - [Querying Vectors](#querying-vectors)
5. [Additional Information](#additional-information)
6. [References and Resources](#references-and-resources)

---

## 1. Introduction <a name="introduction"></a>

Welcome to the documentation for the Swarms `PgVectorVectorStore` class! Swarms is a library that provides various memory and storage options for high-dimensional vectors. In this documentation, we will focus on the `PgVectorVectorStore` class, which is a vector storage driver that uses PostgreSQL with the PGVector extension as the underlying storage engine.

### 1.1 Purpose

The `PgVectorVectorStore` class allows you to interact with a PostgreSQL database and store high-dimensional vectors efficiently. By using Swarms with PostgreSQL and PGVector, you can manage and work with vector data in your applications with ease.

### 1.2 Key Features

- Integration with PostgreSQL and PGVector for vector storage.
- Simple and convenient API for upserting vectors, querying, and loading entries.
- Support for creating and managing vector collections in PostgreSQL.

---

## 2. Overview <a name="overview"></a>

Before diving into the details of the `PgVectorVectorStore` class, let's provide an overview of its purpose and functionality.

The `PgVectorVectorStore` class is designed to:

- Store high-dimensional vectors in a PostgreSQL database with the PGVector extension.
- Offer a seamless and efficient way to upsert vectors into the database.
- Provide methods for loading individual vector entries or all vector entries in a collection.
- Support vector queries, allowing you to find vectors similar to a given query vector.

In the following sections, we will explore the class definition, its parameters, and how to use it effectively.

---

## 3. Class Definition <a name="class-definition"></a>

Let's start by examining the class definition of `PgVectorVectorStore`, including its attributes and parameters.

```python
class PgVectorVectorStore(BaseVectorStore):
    """
    A vector store driver to Postgres using the PGVector extension.

    Attributes:
        connection_string: An optional string describing the target Postgres database instance.
        create_engine_params: Additional configuration params passed when creating the database connection.
        engine: An optional sqlalchemy Postgres engine to use.
        table_name: Optionally specify the name of the table to used to store vectors.
    ...
    """
```

Attributes:

- `connection_string` (Optional[str]): An optional string describing the target Postgres database instance.
- `create_engine_params` (dict): Additional configuration parameters passed when creating the database connection.
- `engine` (Optional[Engine]): An optional SQLAlchemy Postgres engine to use.
- `table_name` (str): Optionally specify the name of the table to be used to store vectors.

### 3.1 Attribute Validators

The class includes validators for the `connection_string` and `engine` attributes to ensure their proper usage. These validators help maintain consistency in attribute values.

### 3.2 Initialization

During initialization, the class checks if an engine is provided. If an engine is not provided, it creates a new database connection using the `connection_string` and `create_engine_params`.

---

## 4. Functionality and Usage <a name="functionality-and-usage"></a>

In this section, we will explore the functionality of the `PgVectorVectorStore` class and provide detailed instructions on how to use it effectively.

### 4.1 Setting Up the Database <a name="setting-up-the-database"></a>

Before using the `PgVectorVectorStore` to store and query vectors, you need to set up the database. This includes creating the necessary extensions and database schema. You can do this using the `setup` method.

```python
def setup(
    self,
    create_schema: bool = True,
    install_uuid_extension: bool = True,
    install_vector_extension: bool = True,
) -> None:
    """
    Provides a mechanism to initialize the database schema and extensions.

    Parameters:
    - create_schema (bool): If True, creates the necessary database schema for vector storage. Default: True.
    - install_uuid_extension (bool): If True, installs the UUID extension in the database. Default: True.
    - install_vector_extension (bool): If True, installs the PGVector extension in the database. Default: True.
    """
```

#### Example 1: Setting Up the Database

```python
# Initialize the PgVectorVectorStore instance
vector_store = PgVectorVectorStore(
    connection_string="your-db-connection-string", table_name="your-table-name"
)

# Set up the database with default settings
vector_store.setup()
```

#### Example 2: Customized Database Setup

```python
# Initialize the PgVectorVectorStore instance
vector_store = PgVectorVectorStore(
    connection_string="your-db-connection-string", table_name="your-table-name"
)

# Set up the database with customized settings
vector_store.setup(
    create_schema=False, install_uuid_extension=True, install_vector_extension=True
)
```

### 4.2 Upserting Vectors <a name="upserting-vectors"></a>

The `upsert_vector` method allows you to insert or update a vector in the collection. You can specify the vector, an optional vector ID, namespace, and metadata.

```python
def upsert_vector(
    self,
    vector: list[float],
    vector_id: Optional[str] = None,
    namespace: Optional[str] = None,
    meta: Optional[dict] = None,
    **kwargs,
) -> str:
    """
    Inserts or updates a vector in the collection.

    Parameters:
    - vector (list[float]): The vector to upsert.
    - vector_id (Optional[str]): An optional ID for the vector. If not provided, a unique ID will be generated.
    - namespace (Optional[str]): An optional namespace for the vector.
    - meta (Optional[dict]): An optional metadata dictionary associated with the vector.
    - **kwargs: Additional keyword arguments.

    Returns:
    - str: The ID of the upserted vector.
    """
```

#### Example: Upserting a Vector

```python
# Initialize the PgVectorVectorStore instance
vector_store = PgVectorVectorStore(
    connection_string="your-db-connection-string", table_name="your-table-name"
)

# Define a vector and upsert it
vector = [0.1, 0.2, 0.3, 0.4]
vector_id = "unique-vector-id"
namespace = "your-namespace"
meta = {"key1": "value1", "key2": "value2"}

vector_store.upsert_vector(
    vector=vector, vector_id=vector_id, namespace=namespace, meta=meta
)
```

### 4.3 Loading Vector Entries <a name="loading-vector-entries"></a>

You can load vector entries from the collection using the `load_entry` and `load_entries` methods.

#### 4

.3.1 Loading a Single Entry

The `load_entry` method allows you to load a specific vector entry based on its identifier and optional namespace.

```python
def load_entry(
    self, vector_id: str, namespace: Optional[str] = None
) -> BaseVectorStore.Entry:
    """
    Retrieves a specific vector entry from the collection based on its identifier and optional namespace.

    Parameters:
    - vector_id (str): The ID of the vector to retrieve.
    - namespace (Optional[str]): An optional namespace for filtering. Default: None.

    Returns:
    - BaseVectorStore.Entry: The loaded vector entry.
    """
```

#### Example: Loading a Single Entry

```python
# Initialize the PgVectorVectorStore instance
vector_store = PgVectorVectorStore(connection_string="your-db-connection-string", table_name="your-table-name")

# Load a specific vector entry
loaded_entry = vector_store.load_entry(vector_id="unique-vector-id", namespace="your-namespace")

if loaded_entry is not None:
    loaded_vector = loaded_entry.vector
    loaded_meta = loaded_entry.meta
    # Use the loaded vector and metadata as needed
else:
    # Vector not found
```

#### 4.3.2 Loading Multiple Entries

The `load_entries` method allows you to load all vector entries from the collection, optionally filtering by namespace.

```python
def load_entries(self, namespace: Optional[str] = None) -> list[BaseVectorStore.Entry]:
    """
    Retrieves all vector entries from the collection, optionally filtering to only those that match the provided namespace.

    Parameters:
    - namespace (Optional[str]): An optional namespace for filtering. Default: None.

    Returns:
    - list[BaseVectorStore.Entry]: A list of loaded vector entries.
    """
```

#### Example: Loading Multiple Entries

```python
# Initialize the PgVectorVectorStore instance
vector_store = PgVectorVectorStore(
    connection_string="your-db-connection-string", table_name="your-table-name"
)

# Load all vector entries in the specified namespace
entries = vector_store.load_entries(namespace="your-namespace")

# Process the loaded entries
for entry in entries:
    vector_id = entry.id
    vector = entry.vector
    meta = entry.meta

    # Handle the loaded entries as needed
```

### 4.4 Querying Vectors <a name="querying-vectors"></a>

You can perform vector queries to find vectors similar to a given query vector using the `query` method. You can specify the query string, the maximum number of results to return, and other options.

```python
def query(
    self,
    query: str,
    count: Optional[int] = BaseVectorStore.DEFAULT_QUERY_COUNT,
    namespace: Optional[str] = None,
    include_vectors: bool = False,
    distance_metric: str = "cosine_distance",
    **kwargs,
) -> list[BaseVectorStore.QueryResult]:
    """
    Performs a search on the collection to find vectors similar to the provided input vector,
    optionally filtering to only those that match the provided namespace.

    Parameters:
    - query (str): The query string to find similar vectors.
    - count (Optional[int]): Maximum number of results to return. Default: BaseVectorStore.DEFAULT_QUERY_COUNT.
    - namespace (Optional[str]): An optional namespace for filtering. Default: None.
    - include_vectors (bool): If True, includes vectors in the query results. Default: False.
    - distance_metric (str): The distance metric to use for similarity measurement.
      Options: "cosine_distance", "l2_distance", "inner_product". Default: "cosine_distance".
    - **kwargs: Additional keyword arguments.

    Returns:
    - list[BaseVectorStore.QueryResult]: A list of query results, each containing vector ID, vector (if included), score, and metadata.
    """
```

#### Example: Querying Vectors

```python
# Initialize the PgVectorVectorStore instance
vector_store = PgVectorVectorStore(
    connection_string="your-db-connection-string", table_name="your-table-name"
)

# Perform a vector query
query_string = "your-query-string"
count = 10  # Maximum number of results to return
namespace = "your-namespace"
include_vectors = False  # Set to True to include vectors in results
distance_metric = "cosine_distance"

results = vector_store.query(
    query=query_string,
    count=count,
    namespace=namespace,
    include_vectors=include_vectors,
    distance_metric=distance_metric,
)

# Process the query results
for result in results:
    vector_id = result.id
    vector = result.vector
    score = result.score
    meta = result.meta

    # Handle the results as needed
```

---

## 5. Additional Information <a name="additional-information"></a>

Here are some additional tips and information for using the `PgVectorVectorStore` class effectively:

- When upserting vectors, you can generate a unique vector ID using a hash of the vector's content to ensure uniqueness.
- Consider using namespaces to organize and categorize vectors within your PostgreSQL database.
- You can choose from different distance metrics (cosine distance, L2 distance, inner product) for vector querying based on your application's requirements.
- Keep your database connection string secure and follow best practices for database access control.

---

## 6. References and Resources <a name="references-and-resources"></a>

Here are some references and resources for further information on Swarms and PostgreSQL with PGVector:

- [Swarms GitHub Repository](https://github.com/swarms): Swarms library on GitHub for updates and contributions.
- [PostgreSQL Official Website](https://www.postgresql.org/): Official PostgreSQL website for documentation and resources.
- [PGVector GitHub Repository](https://github.com/ankane/pgvector): PGVector extension on GitHub for detailed information.

---

This concludes the documentation for the Swarms `PgVectorVectorStore` class. You now have a comprehensive understanding of how to use Swarms with PostgreSQL and PGVector for vector storage. If you have any further questions or need assistance, please refer to the provided references and resources. Happy coding!