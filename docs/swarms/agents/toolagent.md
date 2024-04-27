# ToolAgent Documentation


### Overview and Introduction

The `ToolAgent` class represents an intelligent agent capable of performing a specific task using a pre-trained model and tokenizer. It leverages the Transformer models of the Hugging Face `transformers` library to generate outputs that adhere to a specific JSON schema. This provides developers with a flexible tool for creating bots, text generators, and conversational AI agents. The `ToolAgent` operates based on a JSON schema provided by you, the user. Using the schema, the agent applies the provided model and tokenizer to generate structured text data that matches the specified format.

The primary objective of the `ToolAgent` class is to amplify the efficiency of developers and AI practitioners by simplifying the process of generating meaningful outputs that navigate the complexities of the model and tokenizer.

### Class Definition

The `ToolAgent` class has the following definition:

```python
class ToolAgent(BaseLLM):
    def __init__(
        self,
        name: str,
        description: str,
        model: Any,
        tokenizer: Any,
        json_schema: Any,
        *args,
        **kwargs,
    )
    def run(self, task: str, *args, **kwargs)
    def __call__(self, task: str, *args, **kwargs)
```

### Arguments

The `ToolAgent` class takes the following arguments:

| Argument  | Type | Description |
| --- | --- | --- |
| name  | str  | The name of the tool agent.
| description | str | A description of the tool agent.
| model | Any | The model used by the tool agent (e.g., `transformers.AutoModelForCausalLM`).
| tokenizer | Any | The tokenizer used by the tool agent (e.g., `transformers.AutoTokenizer`).
| json_schema | Any | The JSON schema used by the tool agent.
| *args | - | Variable-length arguments.
| **kwargs | - | Keyword arguments.

### Methods

`ToolAgent` exposes the following methods:

#### `run(self, task: str, *args, **kwargs) -> Any`

- Description: Runs the tool agent for a specific task.
- Parameters:
  - `task` (str): The task to be performed by the tool agent.
  - `*args`: Variable-length argument list.
  - `**kwargs`: Arbitrary keyword arguments.
- Returns: The output of the tool agent.
- Raises: Exception if an error occurs during the execution of the tool agent.


#### `__call__(self, task: str, *args, **kwargs) -> Any`

- Description: Calls the tool agent to perform a specific task.
- Parameters:
  - `task` (str): The task to be performed by the tool agent.
  - `*args`: Variable-length argument list.
  - `**kwargs`: Arbitrary keyword arguments.
- Returns: The output of the tool agent.

### Usage Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from swarms import ToolAgent

# Creating a model and tokenizer
model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b")
tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b")

# Defining a JSON schema
json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
        "is_student": {"type": "boolean"},
        "courses": {"type": "array", "items": {"type": "string"}},
    },
}

# Defining a task
task = "Generate a person's information based on the following schema:"

# Creating the ToolAgent instance
agent = ToolAgent(model=model, tokenizer=tokenizer, json_schema=json_schema)

# Running the tool agent
generated_data = agent.run(task)

# Accessing and printing the generated data
print(generated_data)
```

### Additional Information and Tips

When using the `ToolAgent`, it is important to ensure compatibility between the provided model, tokenizer, and the JSON schema. Additionally, any errors encountered during the execution of the tool agent are propagated as exceptions. Handling such exceptions appropriately can improve the robustness of the tool agent usage.

### References and Resources

For further exploration and understanding of the underlying Transformer-based models and tokenizers, refer to the Hugging Face `transformers` library documentation and examples. Additionally, for JSON schema modeling, you can refer to the official JSON Schema specification and examples.

This documentation provides a comprehensive guide on using the `ToolAgent` class from `swarms` library, and it is recommended to refer back to this document when utilizing the `ToolAgent` for developing your custom conversational agents or text generation tools.
