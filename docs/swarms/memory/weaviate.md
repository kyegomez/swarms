# Weaviate API Client Documentation

## Overview

The Weaviate API Client is an interface to Weaviate, a vector database with a GraphQL API. This client allows you to interact with Weaviate programmatically, making it easier to create collections, add objects, query data, update objects, and delete objects within your Weaviate instance.

This documentation provides a comprehensive guide on how to use the Weaviate API Client, including its initialization, methods, and usage examples.

## Table of Contents

- [Installation](#installation)
- [Initialization](#initialization)
- [Methods](#methods)
  - [create_collection](#create-collection)
  - [add](#add)
  - [query](#query)
  - [update](#update)
  - [delete](#delete)
- [Examples](#examples)

## Installation

Before using the Weaviate API Client, make sure to install the `swarms` library. You can install it using pip:

```bash
pip install swarms
```

## Initialization

To use the Weaviate API Client, you need to initialize an instance of the `WeaviateDB` class. Here are the parameters you can pass to the constructor:

| Parameter            | Type           | Description                                                                                                                      |
|----------------------|----------------|----------------------------------------------------------------------------------------------------------------------------------|
| `http_host`          | str            | The HTTP host of the Weaviate server.                                                                                            |
| `http_port`          | str            | The HTTP port of the Weaviate server.                                                                                            |
| `http_secure`        | bool           | Whether to use HTTPS.                                                                                                            |
| `grpc_host`          | Optional[str]  | The gRPC host of the Weaviate server. (Optional)                                                                                 |
| `grpc_port`          | Optional[str]  | The gRPC port of the Weaviate server. (Optional)                                                                                 |
| `grpc_secure`        | Optional[bool] | Whether to use gRPC over TLS. (Optional)                                                                                         |
| `auth_client_secret` | Optional[Any]  | The authentication client secret. (Optional)                                                                                     |
| `additional_headers` | Optional[Dict[str, str]] | Additional headers to send with requests. (Optional)                                                                          |
| `additional_config`  | Optional[weaviate.AdditionalConfig] | Additional configuration for the client. (Optional)                                                                   |
| `connection_params`  | Dict[str, Any] | Dictionary containing connection parameters. This parameter is used internally and can be ignored in most cases.       |

Here's an example of how to initialize a WeaviateDB:

```python
from swarms.memory import WeaviateDB

weaviate_client = WeaviateDB(
    http_host="YOUR_HTTP_HOST",
    http_port="YOUR_HTTP_PORT",
    http_secure=True,
    grpc_host="YOUR_gRPC_HOST",
    grpc_port="YOUR_gRPC_PORT",
    grpc_secure=True,
    auth_client_secret="YOUR_APIKEY",
    additional_headers={"X-OpenAI-Api-Key": "YOUR_OPENAI_APIKEY"},
    additional_config=None,  # You can pass additional configuration here
)
```

## Methods

### `create_collection`

The `create_collection` method allows you to create a new collection in Weaviate. A collection is a container for storing objects with specific properties.

#### Parameters

- `name` (str): The name of the collection.
- `properties` (List[Dict[str, Any]]): A list of dictionaries specifying the properties of objects to be stored in the collection.
- `vectorizer_config` (Any, optional): Additional vectorizer configuration for the collection. (Optional)

#### Usage

```python
weaviate_client.create_collection(
    name="my_collection",
    properties=[
        {"name": "property1", "dataType": ["string"]},
        {"name": "property2", "dataType": ["int"]},
    ],
    vectorizer_config=None,  # Optional vectorizer configuration
)
```

### `add`

The `add` method allows you to add an object to a specified collection in Weaviate.

#### Parameters

- `collection_name` (str): The name of the collection where the object will be added.
- `properties` (Dict[str, Any]): A dictionary specifying the properties of the object to be added.

#### Usage

```python
weaviate_client.add(
    collection_name="my_collection", properties={"property1": "value1", "property2": 42}
)
```

### `query`

The `query` method allows you to query objects from a specified collection in Weaviate.

#### Parameters

- `collection_name` (str): The name of the collection to query.
- `query` (str): The query string specifying the search criteria.
- `limit` (int, optional): The maximum number of results to return. (Default: 10)

#### Usage

```python
results = weaviate_client.query(
    collection_name="my_collection",
    query="property1:value1",
    limit=20  # Optional, specify the limit

 if needed
)
```

### `update`

The `update` method allows you to update an object in a specified collection in Weaviate.

#### Parameters

- `collection_name` (str): The name of the collection where the object exists.
- `object_id` (str): The ID of the object to be updated.
- `properties` (Dict[str, Any]): A dictionary specifying the properties to update.

#### Usage

```python
weaviate_client.update(
    collection_name="my_collection",
    object_id="object123",
    properties={"property1": "new_value", "property2": 99},
)
```

### `delete`

The `delete` method allows you to delete an object from a specified collection in Weaviate.

#### Parameters

- `collection_name` (str): The name of the collection from which to delete the object.
- `object_id` (str): The ID of the object to delete.

#### Usage

```python
weaviate_client.delete(collection_name="my_collection", object_id="object123")
```

## Examples

Here are three examples demonstrating how to use the Weaviate API Client for common tasks:

### Example 1: Creating a Collection

```python
weaviate_client.create_collection(
    name="people",
    properties=[
        {"name": "name", "dataType": ["string"]},
        {"name": "age", "dataType": ["int"]},
    ],
)
```

### Example 2: Adding an Object

```python
weaviate_client.add(collection_name="people", properties={"name": "John", "age": 30})
```

### Example 3: Querying Objects

```python
results = weaviate_client.query(collection_name="people", query="name:John", limit=5)
```

These examples cover the basic operations of creating collections, adding objects, and querying objects using the Weaviate API Client.

## Additional Information and Tips

- If you encounter any errors during the operations, the client will raise exceptions with informative error messages.
- You can explore more advanced features and configurations in the Weaviate documentation.
- Make sure to handle authentication and security appropriately when using the client in production environments.

## References and Resources

- [Weaviate Documentation](https://weaviate.readthedocs.io/en/latest/): Official documentation for Weaviate.
- [Weaviate GitHub Repository](https://github.com/semi-technologies/weaviate): The source code and issue tracker for Weaviate.

This documentation provides a comprehensive guide on using the Weaviate API Client to interact with Weaviate, making it easier to manage and query your data.