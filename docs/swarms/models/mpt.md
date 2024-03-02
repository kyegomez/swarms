# `MPT7B`
==============================================

The `MPT7B` class is a powerful tool for generating text using pre-trained models. It leverages the `transformers` library from Hugging Face to load models and tokenizers, and to perform the text generation. The class is designed to be flexible and easy to use, with a variety of methods for generating text both synchronously and asynchronously.

## Class Definition
----------------

```
class MPT7B:
    def __init__(self, model_name: str, tokenizer_name: str, max_tokens: int = 100)
    def run(self, task: str, *args, **kwargs) -> str
    async def run_async(self, task: str, *args, **kwargs) -> str
    def generate(self, prompt: str) -> str
    async def generate_async(self, prompt: str) -> str
    def __call__(self, task: str, *args, **kwargs) -> str
    async def __call_async__(self, task: str, *args, **kwargs) -> str
    def batch_generate(self, prompts: list, temperature: float = 1.0) -> list
    def unfreeze_model(self)
```


## Class Parameters
----------------

| Parameter | Type | Description |
| --- | --- | --- |
| `model_name` | str | Name of the pre-trained model to use. |
| `tokenizer_name` | str | Name of the tokenizer to use. |
| `max_tokens` | int | Maximum number of tokens to generate. Default is 100. |

## Class Methods
-------------

| Method | Returns | Description |
| --- | --- | --- |
| `run(task: str, *args, **kwargs)` | str | Run the model with the specified task and arguments. |
| `run_async(task: str, *args, **kwargs)` | str | Run the model asynchronously with the specified task and arguments. |
| `generate(prompt: str)` | str | Generate text from the given prompt. |
| `generate_async(prompt: str)` | str | Generate text asynchronously from the given prompt. |
| `__call__(task: str, *args, **kwargs)` | str | Call the model with the specified task and arguments. |
| `__call_async__(task: str, *args, **kwargs)` | str | Call the model asynchronously with the specified task and arguments. |
| `batch_generate(prompts: list, temperature: float = 1.0)` | list | Generate text for a batch of prompts. |
| `unfreeze_model()` | None | Unfreeze the model for fine-tuning. |

## Usage Examples
--------------

### Example 1: Basic Text Generation

```python
from swarms.models import MPT7B

# Initialize the MPT7B class
mpt = MPT7B("mosaicml/mpt-7b-storywriter", "EleutherAI/gpt-neox-20b", max_tokens=150)

# Generate text
output = mpt.run("generate", "Once upon a time in a land far, far away...")
print(output)
```

### Example 2: Batch Text Generation

```pyton
from swarms.models import MPT7B

# Initialize the MPT7B class
mpt = MPT7B('mosaicml/mpt-7b-storywriter', 'EleutherAI/gpt-neox-20b', max_tokens=150)

# Generate text for a batch of prompts
prompts = ['In the deep jungles,', 'At the heart of the city,']
outputs = mpt.batch_generate(prompts, temperature=0.7)
print(outputs)
```

### Example 3: Asynchronous Text Generation

```python
import asyncio

from swarms.models import MPT7B

# Initialize the MPT7B class
mpt = MPT7B("mosaicml/mpt-7b-storywriter", "EleutherAI/gpt-neox-20b", max_tokens=150)

# Generate text asynchronously
output = asyncio.run(
    mpt.run_async("generate", "Once upon a time in a land far, far away...")
)
print(output)
```

## Additional Information 
----------------------------------

The `batch_generate` method allows for generating text for multiple prompts at once. This can be more efficient than generating text for each prompt individually, especially when working with a large number of prompts.

The `unfreeze_model` method is used to unfreeze the model for fine-tuning. By default, the model parameters are frozen to prevent them from being updated during training. Unfreezing the model allows the parameters to be updated, which can be useful for fine-tuning the model on a specific task.

The `__call__` and `__call_async__` methods are convenience methods that allow the class instance to be called like a function. They simply call the `run` and `run_async` methods, respectively.

## Architecture and Working
------------------------

The `MPT7B` class is designed to be a simple and flexible interface for text generation with pre-trained models. It encapsulates the complexity of loading models and tokenizers, setting up the text generation pipeline, and generating text.

The class uses the `AutoModelForCausalLM` and `AutoTokenizer` classes from the `transformers` library to load the pre-trained model and tokenizer. The `pipeline` function is used to create a text generation pipeline with the loaded model and tokenizer. This pipeline is used to generate text from prompts.

The `run` and `run_async` methods are the main entry points for using the class. They accept a task name and arbitrary arguments, and call the appropriate method based on the task name. The `generate` and `generate_async` methods perform the actual text generation.

The `batch_generate` method allows for generating text for multiple prompts at once. This can be more efficient than generating text for each prompt individually, especially when working with a large number of prompts.

The `unfreeze_model` method is used to unfreeze the model for fine-tuning. By default, the model parameters are frozen to prevent them from being updated during training. Unfreezing the model allows the parameters to be updated, which can be useful for fine-tuning the model on a specific task.

The `__call__` and `__call_async__` methods are convenience methods that allow the class instance to be called like a function. They simply call the `run` and `run_async` methods, respectively.

## Conclusion
----------

The `MPT7B` class provides a powerful and flexible interface for text generation with pre-trained models. It encapsulates the complexity of loading models and tokenizers, setting up the text generation pipeline, and generating text, making it easy to generate high-quality text with just a few lines of code. Whether you're generating text for a single prompt, a batch of prompts, or fine-tuning the model on a specific task, the `MPT7B` class has you covered.