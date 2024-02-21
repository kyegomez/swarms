# limit_tokens_from_string

## Introduction
The `Swarms.utils` library contains utility functions used across codes that handle machine learning and other operations. The `Swarms.utils` library includes a notable function named `limit_tokens_from_string()`. This function particularly limits the number of tokens in a given string. 

# Function: limit_tokens_from_string()
Within the `Swarms.utils` library, there is a method `limit_tokens_from_string(string: str, model: str = "gpt-4", limit: int = 500) -> str:`

## Description
The function `limit_tokens_from_string()` limits the number of tokens in a given string based on the specified threshold. It is primarily useful when you are handling large text data and need to chunk or limit your text to a certain length. Limiting token length could be useful in various scenarios such as when working with data with limited computational resources, or when dealing with models that accept a specific maximum limit of text. 

## Parameters

| Parameter   | Type         | Default Value | Description
| :-----------| :----------- | :------------ | :------------|
| `string`    | `str`        | `None`        | The input string from which the tokens need to be limited. |
| `model`     | `str`        | `"gpt-4"`     | The model used to encode and decode the token. The function defaults to `gpt-4` but you can specify any model supported by `tiktoken`. If a model is not found, it falls back to use `gpt2` |
| `limit`     | `int`        | `500`         | The limit up to which the tokens have to be sliced. Default limit is 500.|

## Returns

| Return      | Type         | Description
| :-----------| :----------- | :------------
| `out`       | `str`        | A string that is constructed back from the encoded tokens that have been limited to a count of `limit` |

## Method Detail and Usage Examples 

The method `limit_tokens_from_string()` takes in three parameters - `string`, `model`, and `limit`. 


First, it tries to get the encoding for the model specified in the `model` argument using `tiktoken.encoding_for_model(model)`. In case the specified model is not found, the function uses `gpt2` model encoding as a fallback.

Next, the input `string` is tokenized using the `encode` method on the `encoding` tensor. This results in the `encoded` tensor.

Then, the function slices the `encoded` tensor to get the first `limit` number of tokens.

Finally, the function converts back the tokens into the string using the `decode` method of the `encoding` tensor. The resulting string `out` is returned.

### Example 1:

```python
from swarms.utils import limit_tokens_from_string

# longer input string
string = "This is a very long string that needs to be tokenized. This string might exceed the maximum token limit, so it will need to be truncated."

# lower token limit
limit = 10

output = limit_tokens_from_string(string, limit=limit)
```

### Example 2:

```python
from swarms.utils import limit_tokens_from_string

# longer input string with different model
string = "This string will be tokenized using gpt2 model. If the string is too long, it will be truncated."

# model
model = "gpt2"

output = limit_tokens_from_string(string, model=model)
```

### Example 3:

```python
from swarms.utils import limit_tokens_from_string

# try with a random model string
string = "In case the method does not find the specified model, it will fall back to gpt2 model."

# model
model = "gpt-4"

output = limit_tokens_from_string(string, model=model)
```

**Note:** If specifying a model not supported by `tiktoken` intentionally, it will fall back to `gpt2` model for encoding.

